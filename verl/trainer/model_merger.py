# model_merger.py
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch
from torch.distributed._tensor import DTensor, Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
    PretrainedConfig,
    PreTrainedModel,
)



import os
import shutil
from pathlib import Path

def reorganize_folders(root_dir: str) -> None:
    """
    重组文件夹结构：
    1. 将actor/huggingface重命名为models并移动到父目录
    2. 删除除新models文件夹外的所有内容
    
    参数:
        root_dir: 最外层目录路径 (示例中的'step_20_reward_0.676')
    """
    root_path = Path(root_dir)
    actor_path = root_path / "actor"
    huggingface_path = actor_path / "huggingface"
    
    # 验证目录结构是否符合预期
    if not actor_path.exists():
        raise FileNotFoundError(f"未找到actor目录: {actor_path}")
    if not huggingface_path.exists():
        raise FileNotFoundError(f"未找到huggingface目录: {huggingface_path}")
    
    # 新models目录路径 (与actor同级)
    models_path = root_path / "models"
    
    print(f"正在将 {huggingface_path} 移动到 {models_path}")
    
    # 移动并重命名huggingface文件夹
    shutil.move(str(huggingface_path), str(models_path))
    
    print("正在清理原始文件...")
    
    # 删除原始actor目录及其内容
    shutil.rmtree(str(actor_path))
    
    # 删除其他可能存在的文件 (根据图片描述)
    for item in root_path.glob("*"):
        if item.name != "models":
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(str(item))
    
    print("文件夹重组完成！")

class ModelMerger:
    def __init__(self):
        pass

    @staticmethod
    def merge_by_placement(tensors: list[torch.Tensor], placement: Placement) -> torch.Tensor:
        """Merge tensors based on their placement type."""
        if placement.is_replicate():
            return tensors[0]
        elif placement.is_partial():
            raise NotImplementedError("Partial placement is not supported yet")
        elif placement.is_shard():
            return torch.cat(tensors, dim=placement.dim).contiguous()
        else:
            raise ValueError(f"Unsupported placement: {placement}")

    def process_model_shards(
        self,
        local_dir: str,
        hf_output_path: Optional[str] = None,
        hf_upload_path: Optional[str] = None
    ) -> None:
        """
        Process and merge distributed model shards into a single model.
        
        Args:
            local_dir: Directory containing the distributed model shards
            hf_output_path: Path to save the merged HuggingFace model (defaults to local_dir/huggingface)
            hf_upload_path: HuggingFace repository path to upload the merged model (optional)
        """
        if hf_output_path is None:
            hf_output_path = os.path.join(local_dir, "huggingface")
        
        assert not local_dir.endswith("huggingface"), "The local_dir should not end with huggingface."

        # Find world size from rank 0 file
        world_size, rank = self._find_world_size_and_rank(local_dir)
        state_dict = self._load_rank_state_dict(local_dir, world_size, rank)
        
        # Determine device mesh configuration
        pivot_key = sorted(state_dict.keys())[0]
        weight = state_dict[pivot_key]
        if isinstance(weight, DTensor):
            device_mesh = weight.device_mesh
            mesh = device_mesh.mesh
            mesh_dim_names = device_mesh.mesh_dim_names
        else:
            mesh = np.array([int(world_size)], dtype=np.int64)
            mesh_dim_names = ("fsdp",)

        print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")
        assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names {mesh_dim_names}."

        # Calculate total shards and mesh shape
        if "tp" in mesh_dim_names:
            total_shards = mesh.shape[-1] * mesh.shape[-2]
            mesh_shape = (mesh.shape[-2], mesh.shape[-1])
        else:
            total_shards = mesh.shape[-1]
            mesh_shape = (mesh.shape[-1],)

        print(f"Processing {total_shards} model shards in total.")
        model_state_dicts = self._load_all_shards(local_dir, world_size, total_shards)
        
        # Merge all shards
        merged_state_dict = self._merge_state_dicts(model_state_dicts, mesh_shape, mesh_dim_names)
        
        # Save merged model
        self._save_merged_model(hf_output_path, merged_state_dict)
        
        # Upload to HuggingFace if specified
        if hf_upload_path:
            self._upload_to_huggingface(hf_output_path, hf_upload_path)

    def _find_world_size_and_rank(self, local_dir: str) -> tuple[int, int]:
        """Find world size and rank from model files."""
        world_size = 0
        rank = 0
        for filename in os.listdir(local_dir):
            match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
            if match:
                world_size = int(match.group(1))
                break

        assert world_size > 0, "No model file with the proper format."
        return world_size, rank

    def _load_rank_state_dict(self, local_dir: str, world_size: int, rank: int) -> dict:
        """Load state dict for a specific rank."""
        rank0_weight_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        return torch.load(rank0_weight_path, map_location="cpu", weights_only=False)

    def _load_all_shards(self, local_dir: str, world_size: int, total_shards: int) -> list[dict]:
        """Load all model shards using thread pool."""
        model_state_dicts = [None] * total_shards
        
        def _load_shard(rank):
            model_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
            return torch.load(model_path, map_location="cpu", weights_only=False)

        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
            futures = []
            for rank in range(total_shards):
                futures.append(executor.submit(_load_shard, rank))
            
            for rank, future in enumerate(futures):
                model_state_dicts[rank] = future.result()
        
        return model_state_dicts

    def _merge_state_dicts(
        self, 
        model_state_dicts: list[dict], 
        mesh_shape: tuple, 
        mesh_dim_names: tuple[str]
    ) -> dict[str, torch.Tensor]:
        """Merge all state dicts into one."""
        state_dict = {}
        param_placements = {}
        keys = set(model_state_dicts[0].keys())
        
        # Collect all tensors for each key
        for key in keys:
            state_dict[key] = []
            for model_state_dict in model_state_dicts:
                tensor = model_state_dict[key]
                
                if isinstance(tensor, DTensor):
                    state_dict[key].append(tensor._local_tensor.bfloat16())
                    placements = tuple(tensor.placements)
                    # replicated placement at ddp dimension can be discarded
                    if mesh_dim_names[0] == "ddp":
                        placements = placements[1:]

                    if key not in param_placements:
                        param_placements[key] = placements
                    else:
                        assert param_placements[key] == placements
                else:
                    state_dict[key].append(tensor.bfloat16())

        # Merge collected tensors
        for key in sorted(state_dict):
            if not isinstance(state_dict[key], list):
                print(f"No need to merge key {key}")
                continue

            if key in param_placements:
                # merge shards
                placements = param_placements[key]
                if len(mesh_shape) == 1:
                    # 1-D list, FSDP without TP
                    assert len(placements) == 1
                    shards = state_dict[key]
                    state_dict[key] = self.merge_by_placement(shards, placements[0])
                else:
                    # 2-D list, FSDP + TP
                    raise NotImplementedError("FSDP + TP is not supported yet.")
            else:
                state_dict[key] = torch.cat(state_dict[key], dim=0)
        
        return state_dict

    def _save_merged_model(self, output_path: str, state_dict: dict[str, torch.Tensor]) -> None:
        """Save the merged model in HuggingFace format."""
        print(f"Saving model to {output_path}...")
        config = AutoConfig.from_pretrained(output_path)
        architectures = getattr(config, "architectures", ["Unknown"])

        if "ForTokenClassification" in architectures[0]:
            AutoClass = AutoModelForTokenClassification
        elif "ForConditionalGeneration" in architectures[0]:
            AutoClass = AutoModelForImageTextToText
        elif "ForCausalLM" in architectures[0]:
            AutoClass = AutoModelForCausalLM
        else:
            raise NotImplementedError(f"Unknown architecture {architectures}.")

        with torch.device("meta"):
            model = AutoClass.from_config(config, torch_dtype=torch.bfloat16)

        assert isinstance(model, PreTrainedModel)
        model.to_empty(device="cpu")
        model.load_state_dict(state_dict)
        model.save_pretrained(output_path)

    def _upload_to_huggingface(self, local_path: str, remote_path: str) -> None:
        """Upload model to HuggingFace Hub."""
        from huggingface_hub import HfApi

        print(f"Uploading model to {remote_path}...")
        api = HfApi()
        api.create_repo(repo_id=remote_path, private=False, exist_ok=True)
        api.upload_folder(repo_id=remote_path, folder_path=local_path, repo_type="model")


def merge_model_shards(
    local_dir: str,
    hf_output_path: Optional[str] = None,
    hf_upload_path: Optional[str] = None
) -> None:
    """
    Convenience function to merge model shards.
    
    Args:
        local_dir: Directory containing the distributed model shards
        hf_output_path: Path to save the merged HuggingFace model (defaults to local_dir/huggingface)
        hf_upload_path: HuggingFace repository path to upload the merged model (optional)
    """
    merger = ModelMerger()
    merger.process_model_shards(local_dir, hf_output_path, hf_upload_path)
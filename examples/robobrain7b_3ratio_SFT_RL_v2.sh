#!/bin/bash
# REMINDER: this script uses test data split and should ONLY be used for debugging. DO NOT use for training.

set -x

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
MODEL_PATH=/mnt/lyc/wuxinrui/EasyR1/checkpoints/robobrain7b_3ratio_SFT_RL_v1/robobrain7b_3ratio_SFT_RL_v1/robobrain7b_3ratio_SFT_RL_v1_global_step_33/models  # replace it with your local file path
export budget=150
export stage=2
export remaining=3ratio

Version_RL=robobrain7b_3ratio_SFT_RL_v2


python3 -m verl.trainer.main \
    config=examples/${Version_RL}.yaml \
    data.train_files=/mnt/data/wuxinrui/BAAI_cot_sft_data/RL_Train_data/choices_datasets.parquet \
    data.val_files=/mnt/data/wuxinrui/BAAI_cot_sft_data/RL_Test_data/ERQA/data/test-00000-of-00001_val_RL.parquet \
    data.rollout_batch_size=256 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.limit_images=10 \
    worker.rollout.budget=${budget} \
    worker.rollout.stage=${stage} \
    trainer.project_name=${Version_RL} \
    trainer.experiment_name=${Version_RL} \
    trainer.n_gpus_per_node=4 \

#!/bin/bash
# REMINDER: this script uses test data split and should ONLY be used for debugging. DO NOT use for training.
#g 使用/data2/wuxinrui/R1-training-MLLM/checkpoints/robobrain7b_3ratio_SFT_RL_v1/robobrain7b_3ratio_SFT_RL_v7/robobrain7b_3ratio_SFT_RL_v7_step_17_reward_0.43859649122807015/models从budget150开始续训
set -x

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
MODEL_PATH=/data2/wuxinrui/R1-training-MLLM/checkpoints/robobrain7b_3ratio_SFT_RL_v1/robobrain7b_3ratio_SFT_RL_v7/robobrain7b_3ratio_SFT_RL_v7_step_17_reward_0.43859649122807015/models 
# replace it with your local file path
PARENT_DIR=$(dirname "$MODEL_PATH")
MODEL_NAME=$(basename "$PARENT_DIR")

export stage=1
export remaining=3ratio

Version_RL=robobrain7b_3ratio_SFT_RL_v8


python3 -m verl.trainer.main \
    config=examples/${Version_RL}.yaml \
    data.train_files=/data2/wuxinrui/BAAI_cot_sft_data/RL_Train_data/choices_datasets_withoutCOT_2k.parquet \
    data.val_files=/data2/wuxinrui/BAAI_cot_sft_data/RL_Test_data/data/test-00000-of-00001_val_RL.parquet \
    data.rollout_batch_size=256 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.limit_images=10 \
    worker.rollout.stage=${stage} \
    trainer.experiment_name=${Version_RL} \
    trainer.n_gpus_per_node=4 \
    trainer.load_checkpoint_path=null \
    trainer.val_data_save_folder=eval_samples/${Version_RL}_${MODEL_NAME}_stage${stage}_remaining${remaining}_eval \
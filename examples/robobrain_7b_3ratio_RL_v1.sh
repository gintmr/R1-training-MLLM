#!/bin/bash

set -x
export CUDA_VISIBLE_DEVICES=1,2,3,4

export PYTHONUNBUFFERED=1
export steady=robobrain_7b_3ratio_RL_v1
export TENSORBOARD_DIR=tensorlog_${steady}
MODEL_PATH=/mnt/lyc/wuxinrui/RoboBrain-2/HF_Models/BAAI-RoboBrain2.0-7B

python3 -m verl.trainer.main \
    config=/mnt/lyc/wuxinrui/EasyR1/examples/robobrain_7b_3ratio_RL_v1.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=4 \


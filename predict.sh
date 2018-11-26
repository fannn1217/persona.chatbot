#!/usr/bin/env bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=0
python3 udc_predict.py --model_dir=./runs/1480971822/
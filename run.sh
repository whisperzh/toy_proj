#!/bin/bash
# run.sh

# 启动中间层
./build/nccl_dynamic > middle.log 2>&1 &

# 启动 PyTorch 程序
python3 worker.py
#!/bin/bash

WORLD_SIZE=2
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

torchrun \
  --nproc_per_node=$WORLD_SIZE \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  main.py

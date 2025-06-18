#!/bin/bash

WORLD_SIZE=2
MASTER_ADDR=3.136.159.33
MASTER_PORT=29500

torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  main.py

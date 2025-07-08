#!/bin/bash

RANK=0 WORLD_SIZE=2 LOCAL_RANK=0 \
INIT_METHOD=tcp://<主节点公网/内网IP>:23456 \
NCCL_SOCKET_IFNAME=ens5 \
NCCL_IB_DISABLE=1 \
NCCL_DEBUG=INFO \
python3 main.py

# torchrun \
#   --nproc_per_node=1 \
#   --nnodes=2 \
#   --node_rank=0 \
#   --master_addr=$MASTER_ADDR \
#   --master_port=$MASTER_PORT \
#   main.py

#!/bin/bash

#########################
# ✅ 用户需配置的变量
#########################

# 当前节点的 RANK（0 或 1）
export RANK=0

# 世界总节点数（两台机器）
export WORLD_SIZE=2

# 当前节点使用的 GPU ID
export LOCAL_RANK=0

# 主节点的私网 IP（确保可达，不能用公网 IP）
MASTER_ADDR=172.31.46.106
MASTER_PORT=23456

#########################
# ✅ NCCL 环境变量（针对 G5 TCP 优化）
#########################

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ens5   # G5 默认网卡名称，确保用内网
export NCCL_IB_DISABLE=1         # 显式禁用 InfiniBand
export NCCL_NET=Socket           # 明确指定用 Socket 网络
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=4
export NCCL_MIN_NCHANNELS=4

#########################
# ✅ 启动脚本（INIT_METHOD 为主节点地址）
#########################

export INIT_METHOD=tcp://$MASTER_ADDR:$MASTER_PORT

echo "Starting NCCL TCP Test with the following settings:"
echo "RANK=$RANK, WORLD_SIZE=$WORLD_SIZE, LOCAL_RANK=$LOCAL_RANK"
echo "INIT_METHOD=$INIT_METHOD"
echo "----------------------------------------"

python3 main.py

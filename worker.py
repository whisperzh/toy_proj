# File: worker.py
import os
import time
import json
import torch
import torch.distributed as dist

def load_config():
    with open("config.json") as f:
        cfg = json.load(f)
    return cfg["rank"], cfg["world_size"], cfg["master_ip"]

def init_dist():
    rank, world_size, master_ip = load_config()
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = master_ip
    os.environ["MASTER_PORT"] = "23456"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"[Rank {rank}] Initialized.")

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
        print(f"[Rank {dist.get_rank()}] Destroyed group.")

def training_loop():
    rank = dist.get_rank()
    for i in range(1000):
        tensor = torch.ones(1).cuda()
        dist.all_reduce(tensor)
        print(f"[Rank {rank}] Step {i}, Tensor = {tensor.item()}")
        time.sleep(2)

        if os.path.exists("reload.flag"):
            print(f"[Rank {rank}] Detected reload.flag, reloading...")
            os.remove("reload.flag")
            cleanup()
            init_dist()

if __name__ == "__main__":
    init_dist()
    training_loop()

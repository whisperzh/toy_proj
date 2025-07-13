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
        rank = dist.get_rank()
        dist.destroy_process_group()
        print(f"[Rank {rank}] Destroyed group.")

def training_loop():
    rank = dist.get_rank()
    for i in range(1000):
        tensor =  torch.ones(1024 * 1024, device='cuda')
        
        #测时间
        torch.cuda.synchronize()
        start = time.time()
        
        dist.all_reduce(tensor)
        
        #测时间结束
        torch.cuda.synchronize()
        end = time.time()
        
        latency_ms = (end - start) * 1000
        print(f"[Rank {rank}] Step {i}, Tensor = {tensor.shape}, AllReduce took {latency_ms:.3f} ms")
        
        time.sleep(2)

        # 所有rank通信完成，准备检测reload
        dist.barrier()

        # 💡 只有rank 0读取 reload.flag，其余等它广播决定
        if rank == 0:
            need_reload = int(os.path.exists("reload.flag"))
            if need_reload:
                os.remove("reload.flag")
        else:
            need_reload = 0

        # 所有rank都使用rank 0的判断结果
        need_reload_tensor = torch.tensor([need_reload], device="cuda")
        dist.broadcast(need_reload_tensor, src=0)

        dist.barrier()

        if need_reload_tensor.item() == 1:
            print(f"[Rank {rank}] Reloading distributed group...")
            cleanup()
            time.sleep(1)
            init_dist()

if __name__ == "__main__":
    init_dist()
    training_loop()

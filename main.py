import os
import torch
import torch.distributed as dist
import time

def init_process_group():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 指定当前进程的 GPU
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method=os.environ["INIT_METHOD"],  # 如 tcp://<主节点IP>:23456
        world_size=world_size,
        rank=rank
    )

    return rank, local_rank

def measure_send_recv(rank, local_rank):
    device = torch.device(f"cuda:{local_rank}")
    tensor_size = 1024 * 1024 * 100  # 100MB
    send_tensor = torch.randn(tensor_size, dtype=torch.float32, device=device)
    recv_tensor = torch.empty_like(send_tensor)

    # 确保 CUDA 操作完成
    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        # Rank 0 发送 tensor 给 Rank 1
        start_time = time.time()
        dist.send(send_tensor, dst=1)
        torch.cuda.synchronize()
        send_duration = time.time() - start_time
        print(f"[Rank 0] Send done in {send_duration*1000:.2f} ms")

        # 然后接收回来
        start_time = time.time()
        dist.recv(recv_tensor, src=1)
        torch.cuda.synchronize()
        recv_duration = time.time() - start_time
        print(f"[Rank 0] Recv done in {recv_duration*1000:.2f} ms")

    elif rank == 1:
        dist.recv(recv_tensor, src=0)
        torch.cuda.synchronize()

        dist.send(recv_tensor, dst=0)
        torch.cuda.synchronize()

if __name__ == "__main__":
    rank, local_rank = init_process_group()
    measure_send_recv(rank, local_rank)

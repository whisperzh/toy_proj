import os
import torch
import torch.distributed as dist
import time

def init_process_group():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method=os.environ["INIT_METHOD"],
        world_size=world_size,
        rank=rank
    )

    print(f"[Rank {rank}] Initialized process on cuda:{local_rank}")
    return rank, local_rank

def measure_send_recv(rank, local_rank, n_iters=5, tensor_size_mb=100):
    device = torch.device(f"cuda:{local_rank}")
    tensor_size = 1024 * 1024 * tensor_size_mb  # 转换为 float32 个数
    send_tensor = torch.randn(tensor_size, dtype=torch.float32, device=device)
    recv_tensor = torch.empty_like(send_tensor)

    send_times = []
    recv_times = []

    for i in range(n_iters):
        # 同步所有进程
        dist.barrier()
        torch.cuda.synchronize()

        if rank == 0:
            # 发送
            start = time.time()
            dist.send(send_tensor, dst=1)
            torch.cuda.synchronize()
            send_duration = time.time() - start

            # 接收回来
            start = time.time()
            dist.recv(recv_tensor, src=1)
            torch.cuda.synchronize()
            recv_duration = time.time() - start

            send_times.append(send_duration)
            recv_times.append(recv_duration)

            print(f"[Rank 0][Iter {i}] Send: {send_duration*1000:.2f} ms | Recv: {recv_duration*1000:.2f} ms")

        elif rank == 1:
            dist.recv(recv_tensor, src=0)
            torch.cuda.synchronize()
            dist.send(recv_tensor, dst=0)
            torch.cuda.synchronize()

    if rank == 0:
        avg_send = sum(send_times) / n_iters
        avg_recv = sum(recv_times) / n_iters
        total_avg = avg_send + avg_recv
        total_data_mb = tensor_size_mb * 2  # 往返
        print(f"\n[Rank 0] ===== Final Summary =====")
        print(f"Tensor size: {tensor_size_mb} MB")
        print(f"Average Send: {avg_send*1000:.2f} ms")
        print(f"Average Recv: {avg_recv*1000:.2f} ms")
        print(f"Round-trip latency: {total_avg*1000:.2f} ms")
        print(f"Effective bandwidth: {total_data_mb / total_avg / 1024:.2f} GB/s")

if __name__ == "__main__":
    rank, local_rank = init_process_group()
    measure_send_recv(rank, local_rank)

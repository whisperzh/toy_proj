import os
import torch
import torch.distributed as dist
from datetime import timedelta

def main():
    # Get rank and world size from environment
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Set the correct GPU
    torch.cuda.set_device(local_rank)

    # Initialize the process group
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=10)
    )

    # Create a tensor filled with the rank value
    tensor = torch.ones(1).cuda() * rank
    print(f"[Rank {rank}] Before all_reduce: {tensor.item()}")

    # All-reduce: sum tensor across all ranks
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"[Rank {rank}] After all_reduce: {tensor.item()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

# test_nccl.py
import torch
import os

def main():
    torch.distributed.init_process_group(backend='nccl')
    print(f"Rank {torch.distributed.get_rank()}: NCCL initialized")

if __name__ == "__main__":
    main()
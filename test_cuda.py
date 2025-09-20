# cuda_test.py
import torch

def main():
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

if __name__ == "__main__":
    main()
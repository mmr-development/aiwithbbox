import torch

def main():
    print("PyTorch version:", torch.__version__)
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)
    if cuda_available:
        print("Number of GPUs:", torch.cuda.device_count())
        print("Current GPU device:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("No CUDA-capable GPU detected or CUDA drivers not installed.")

if __name__ == "__main__":
    main()
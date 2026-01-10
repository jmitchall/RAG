# CUDA
## Step 1: Check if you have a compatible Nvidia GPU and CUDA installed

    nvidia-smi

## Step 2: Check if CUDA Toolkit is Installed and CUDA_HOME is Set

Even though the driver is present, the CUDA Toolkit (which provides the compiler and libraries) must also be installed, and the CUDA_HOME environment variable must point to it.

The CUDA Toolkit is required to build and run GPU-accelerated Python packages like vLLM.

## Step 3: Download and Install the CUDA Toolkit

Go to the official CUDA Toolkit download page:
https://developer.nvidia.com/cuda-downloads

Select:

- Operating System: Windows
- Architecture: x86_64
- Version: Windows 11
- Installer Type: exe (local or network)

Download the version that matches your driver (CUDA 13.0 is recommended).

Run the installer and follow the prompts. Accept defaults unless you have a specific need to change them.

After installation, restart your computer.

## Step 4: Verify CUDA Toolkit Installation `nvcc`

1. Open PowerShell and run:

        dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"

    You should now see a folder like v13.0 or similar.

2. Next, check if the CUDA compiler (`nvcc`) is available:

        & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe" --version

    (The & at the start is required in PowerShell.)

## Step 5: Set the `CUDA_HOME` Environment Variable

Open PowerShell as Administrator and run:

    [Environment]::SetEnvironmentVariable("CUDA_HOME", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0", "User")

Or, for the current session:

    $env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"

## Step 6: Allow Script Execution (if needed)

    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
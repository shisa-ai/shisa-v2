# Inspired by vLLM's collect_env.py

import subprocess
import sys
from collections import namedtuple
from typing import Any, Dict, Optional

def run_command(command: str) -> Optional[str]:
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(command.split(),
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None

def get_package_version(package_name: str) -> str:
    """Get version of an installed package or 'Not installed' if not found."""
    try:
        module = __import__(package_name)
        return getattr(module, '__version__', 'Version not found')
    except ImportError:
        return 'Not installed'

def get_cuda_version() -> Optional[str]:
    """Get CUDA version from nvcc or None if not found."""
    return run_command('nvcc --version')

def get_rocm_version() -> Optional[str]:
    """Get ROCm version or None if not found."""
    return run_command('rocm-smi --version')

class MLEnvironment:
    """Class to collect and display ML environment information."""
    
    def __init__(self):
        self.info: Dict[str, Any] = {}
        
        # Define packages to check
        self.packages = [
            'torch',
            'torchao',
            'flash_attn',
            'deepspeed',
            'accelerate',
            'transformers',
            'bitsandbytes',
            'axolotl',
            'torchtune',
        ]
        
        self._collect_info()
    
    def _collect_info(self) -> None:
        """Collect all environment information."""
        # Get GPU information
        self.info['CUDA'] = get_cuda_version()
        self.info['ROCm'] = get_rocm_version()
        
        # Check if PyTorch has CUDA/ROCm support
        try:
            import torch
            self.info['PyTorch CUDA Available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                self.info['PyTorch CUDA Version'] = torch.version.cuda
                self.info['GPU Count'] = torch.cuda.device_count()
                self.info['GPU Names'] = [torch.cuda.get_device_name(i) 
                                        for i in range(torch.cuda.device_count())]
            
            # Check for ROCm support
            self.info['PyTorch HIP Version'] = getattr(torch.version, 'hip', None)
        except ImportError:
            self.info['PyTorch CUDA Available'] = None
            self.info['PyTorch CUDA Version'] = None
            self.info['GPU Count'] = None
            self.info['GPU Names'] = None
            self.info['PyTorch HIP Version'] = None
        
        # Get package versions
        self.info['Package Versions'] = {
            package: get_package_version(package)
            for package in self.packages
        }
    
    def __str__(self) -> str:
        """Format environment information for display."""
        output = []
        
        # GPU Information
        output.append("=== GPU Information ===")
        output.append(f"CUDA: {self.info['CUDA'] or 'Not found'}")
        output.append(f"ROCm: {self.info['ROCm'] or 'Not found'}")
        output.append(f"PyTorch CUDA Available: {self.info['PyTorch CUDA Available']}")
        output.append(f"PyTorch CUDA Version: {self.info['PyTorch CUDA Version'] or 'N/A'}")
        output.append(f"PyTorch HIP Version: {self.info['PyTorch HIP Version'] or 'N/A'}")
        
        if self.info['GPU Count']:
            output.append(f"\nGPU Count: {self.info['GPU Count']}")
            for i, name in enumerate(self.info['GPU Names']):
                output.append(f"GPU {i}: {name}")
        
        # Package Versions
        output.append("\n=== Package Versions ===")
        for package, version in self.info['Package Versions'].items():
            output.append(f"{package}: {version}")
        
        return "\n".join(output)

def main():
    env = MLEnvironment()
    print(env)

if __name__ == "__main__":
    main()

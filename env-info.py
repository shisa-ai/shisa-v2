# Inspired by vLLM's collect_env.py

import platform
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

def get_system_info() -> Dict[str, Any]:
    """Collect system information using various methods."""
    info = {}
    
    '''
    # Try inxi first for comprehensive system info
    inxi_output = run_command('inxi')
    if inxi_output:
        info['inxi_output'] = inxi_output
        return {'inxi_output': inxi_output}
    '''
    
    # Fallback to individual commands if inxi isn't available
    info = {
        'os_info': None,
        'kernel': None,
        'cpu_info': None,
        'memory_info': None
    }
    
    # OS Information
    try:
        # Try lsb_release first
        lsb_output = run_command('lsb_release -d')
        if lsb_output:
            info['os_info'] = lsb_output.split(':')[1].strip()
    except Exception:
        pass

    # Kernel Information
    try:
        info['kernel'] = run_command('uname -a')
    except Exception:
        pass

    # CPU Information
    try:
        if platform.system() == 'Linux':
            # Try to get detailed CPU info from /proc/cpuinfo
            with open('/proc/cpuinfo', 'r') as f:
                cpu_info = f.read()
            
            # Parse for relevant information
            processor_count = cpu_info.count('processor\t:')
            model_name = [line for line in cpu_info.split('\n') if 'model name' in line]
            if model_name:
                model_name = model_name[0].split(':')[1].strip()
            else:
                model_name = platform.processor()
                
            info['cpu_info'] = f"CPU: {model_name} (x{processor_count})"
        else:
            info['cpu_info'] = f"CPU: {platform.processor()}"
    except Exception:
        pass

    # Memory Information
    try:
        if platform.system() == 'Linux':
            with open('/proc/meminfo', 'r') as f:
                mem_info = f.read()
            
            # Parse total memory
            total_memory = [line for line in mem_info.split('\n') if 'MemTotal' in line]
            if total_memory:
                total_memory = int(total_memory[0].split()[1]) // 1024  # Convert to MB
                info['memory_info'] = f"Total Memory: {total_memory} MB"
    except Exception:
        pass

    return info

class MLEnvironment:
    """Class to collect and display ML environment information."""
    
    def __init__(self):
        self.info: Dict[str, Any] = {}
        
        # Define packages to check
        self.packages = [
            'triton',
            'torch',
            'torchao',
            'transformers',
            'flash_attn',
            'xformers',
            'deepspeed',
            'accelerate',
            'bitsandbytes',
            'axolotl',
            'torchtune',
        ]
        
        self._collect_info()
    
    def _collect_info(self) -> None:
        """Collect all environment information."""
        # Get system information
        self.info['system'] = get_system_info()
        
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
        
        # System Information
        output.append("=== System Information ===")
        if self.info['system'].get('inxi_output'):
            output.append(self.info['system']['inxi_output'])
        else:
            for key, value in self.info['system'].items():
                if value:
                    output.append(f"{key.replace('_', ' ').title()}: {value}")
        
        # GPU Information
        output.append("\n=== GPU Information ===")
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

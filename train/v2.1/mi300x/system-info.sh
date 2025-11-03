#!/bin/bash
echo "=== AMD Dev Node System Information ==="
echo "Hostname: $(hostname)"
echo "OS: $(lsb_release -d | cut -f2)"
echo "Kernel: $(uname -r)"
echo "Uptime: $(uptime -p)"
echo "Memory: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
echo "Disk: $(df -h / | tail -1 | awk '{print $3 "/" $2 " (" $5 " used)"}')"
if command -v node &> /dev/null; then
    echo "Node.js: $(node --version)"
    echo "npm: $(npm --version)"
fi
if command -v conda &> /dev/null; then
    echo "Conda: $(conda --version)"
fi
if command -v mamba &> /dev/null; then
    echo "Mamba: $(mamba --version | head -1)"
fi
if command -v starship &> /dev/null; then
    echo "Starship: $(starship --version)"
fi
if command -v atuin &> /dev/null; then
    echo "Atuin: $(atuin --version)"
fi

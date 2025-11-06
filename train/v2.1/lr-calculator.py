import math
from tabulate import tabulate

def calculate_scaled_lr(known_lr: float, known_params: float, target_params: float) -> float:
    """
    Calculates the estimated optimal learning rate for a target model size
    based on a known optimal LR for a different size, using the 1/sqrt(N) scaling law.

    Assumes other training parameters (dataset, steps, global batch size) remain constant.

    Args:
        known_lr: The empirically found optimal learning rate for the known model.
        known_params: The number of parameters in the known model (e.g., 8e9 for 8B).
                    Use scientific notation like XeY for billions (e.g., 8e9).
        target_params: The number of parameters in the target model (e.g., 70e9 for 70B).
                       Use scientific notation like XeY for billions.

    Returns:
        The estimated optimal learning rate for the target model size.

    Raises:
        ValueError: If known_params or target_params is less than or equal to 0.
    """
    if known_params <= 0:
        raise ValueError("Known model parameters must be positive.")
    if target_params <= 0:
        raise ValueError("Target model parameters must be positive.")

    # Apply the scaling law: LR_new = LR_known * sqrt(N_known / N_new)
    scaling_factor = math.sqrt(known_params / target_params)
    scaled_lr = known_lr * scaling_factor

    return scaled_lr


def format_param_size(params: float) -> str:
    """Format parameter count as readable string"""
    if params >= 1e9:
        return f"{params/1e9:.1f}B" if params/1e9 != int(params/1e9) else f"{int(params/1e9)}B"
    elif params >= 1e6:
        return f"{params/1e6:.0f}M"
    else:
        return f"{params:.0f}"


def calculate_lr_table(sft_lr: float, dpo_lr: float, known_params: float, target_sizes: list):
    """
    Calculate learning rates for multiple model sizes and return as table data.
    
    Args:
        sft_lr: Known optimal SFT learning rate
        dpo_lr: Known optimal DPO learning rate  
        known_params: The number of parameters in the known model
        target_sizes: List of target parameter counts
        
    Returns:
        List of [size, sft_lr, dpo_lr] for tabulate
    """
    table_data = []
    
    for params in target_sizes:
        size_str = format_param_size(params)
        scaled_sft_lr = calculate_scaled_lr(sft_lr, known_params, params)
        scaled_dpo_lr = calculate_scaled_lr(dpo_lr, known_params, params)
        
        table_data.append([size_str, f"{scaled_sft_lr:.2e}", f"{scaled_dpo_lr:.2e}"])
    
    return table_data


def get_common_sizes():
    """Returns a list of common model parameter counts"""
    return [
        0.27e9,   # Gemma-3 270M
        0.6e9,    # Qwen3 0.6B
        1e9,      # 1B
        1.2e9,    # 1.2B
        3e9,      # 3B
        3.84e9,   # 3.8B
        4e9,   # 4B
        8e9,      # Llama 3 8B
        12e9,     # Llama 3 12B
        14e9,     # Llama 3 14B
        24e9,     # Llama 3 24B
        27e9,     # Llama 3 27B
        32e9,     # Llama 3 32B
        70e9,     # Llama 3 70B
        405e9,    # Llama 3 405B
    ]

if __name__ == "__main__":
    # Known optimal learning rates for Llama 3 8B (from 022/024 scripts)
    llama3_8b_params = 8e9  # 8 billion parameters
    sft_lr = 1e-5     # From 022-llama3.1-8b-v2new.openrlhf.sh
    dpo_lr = 1.25e-7  # From 024-llama3.1-8b-v2new-dpo405b.openrlhf.sh
    
    # Get model sizes and calculate learning rates
    target_sizes = get_common_sizes()
    table_data = calculate_lr_table(sft_lr, dpo_lr, llama3_8b_params, target_sizes)
    
    # Print table
    headers = ["Size", "SFT LR", "DPO LR"]
    print("Learning Rate Scaling (Based on Llama 3 8B)")
    print("-" * 40)
    print(tabulate(table_data, headers=headers, tablefmt="simple"))
    
    # Highlight Gemma-3 270M specifically
    gemma_270m_params = 0.27e9
    sft_lr_270m = calculate_scaled_lr(sft_lr, llama3_8b_params, gemma_270m_params)
    dpo_lr_270m = calculate_scaled_lr(dpo_lr, llama3_8b_params, gemma_270m_params)
    
    print(f"\n🎯 Recommended for Gemma-3 270M:")
    print(f"   SFT Learning Rate: {sft_lr_270m:.2e}")
    print(f"   DPO Learning Rate: {dpo_lr_270m:.2e}")

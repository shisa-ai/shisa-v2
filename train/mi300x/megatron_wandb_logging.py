"""Monkeypatch Megatron-LM training logging to emit richer W&B metrics."""

import math
import os
import sys
from typing import Dict

import torch

# Ensure Megatron-LM is on sys.path when this module is imported from outside the repo tree
_MEGATRON_PATH = os.environ.get("MEGATRON_LM_PATH", "/workspace/Megatron-LM")
if _MEGATRON_PATH not in sys.path:
    sys.path.insert(0, _MEGATRON_PATH)

from megatron.training import training as training_module
from megatron.training.training import (
    get_args,
    get_timers,
    get_tensorboard_writer,
    get_wandb_writer,
    get_one_logger,
    num_floating_point_operations,
    num_floating_point_operations_mla_moe,
    get_num_microbatches,
    report_theoretical_memory,
    report_memory,
    print_rank_last,
)
from megatron.training.utils import is_last_rank
from megatron.training.training import one_logger_utils
from megatron.core import mpu
from megatron.core.transformer.moe import moe_utils as moe_utils_module
from megatron.core.transformer.moe.moe_utils import (
    reduce_aux_losses_tracker_across_ranks,
    clear_aux_losses_tracker,
)
from megatron.core import parallel_state

# Keep references to original functions so we can delegate when needed.
_original_training_log = training_module.training_log
_original_track_moe_metrics = moe_utils_module.track_moe_metrics

# Storage for the latest per-iteration MoE metrics so the patched training_log can attach
# them to the consolidated W&B payload.
_LAST_MOE_METRICS: Dict[str, float] = {}


def _maybe_item(value):
    """Convert torch tensors to Python floats for logging."""
    if isinstance(value, (float, int)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:  # pragma: no cover - fall back to best effort
            pass
    return float(value)


def patched_track_moe_metrics(
    loss_scale,
    iteration,
    writer,
    wandb_writer=None,  # ignored; logging handled centrally in patched training_log
    total_loss_dict=None,
    per_layer_logging=False,
):
    """Patch Megatron's MoE metric tracker to capture per-iteration summaries."""
    reduce_aux_losses_tracker_across_ranks()
    tracker = parallel_state.get_moe_layer_wise_logging_tracker()

    metrics: Dict[str, float] = {}

    if writer is not None:
        aux_losses = {k: v['values'].float() * loss_scale for k, v in tracker.items()}
        for name, loss_list in aux_losses.items():
            mean_val = loss_list.mean()
            metrics[f"moe/{name}"] = _maybe_item(mean_val)

            if total_loss_dict is not None:
                if name not in total_loss_dict:
                    total_loss_dict[name] = loss_list.mean()
                else:
                    total_loss_dict[name] += loss_list.mean()

            writer.add_scalar(name, mean_val, iteration)
            if per_layer_logging:
                for layer_idx, layer_val in enumerate(loss_list.tolist()):
                    writer.add_scalar(f"moe/{name}_layer_{layer_idx}", layer_val, iteration)

    clear_aux_losses_tracker()

    # Record metrics so training_log can include them in the consolidated W&B payload.
    _LAST_MOE_METRICS.clear()
    _LAST_MOE_METRICS.update(metrics)

    # Preserve original return behaviour (None) while still providing compatibility if
    # downstream code inspects the return value.
    return metrics or None


def patched_training_log(
    loss_dict,
    total_loss_dict,
    learning_rate,
    decoupled_learning_rate,
    iteration,
    loss_scale,
    report_memory_flag,
    skipped_iter,
    grad_norm,
    params_norm,
    num_zeros_in_grad,
):
    """Wrapper around Megatron's training_log to enrich W&B metrics."""

    args = get_args()
    timers = get_timers()

    # Predict the total iteration count that the original function will see once it updates
    # the accumulators. We need this to compute iteration-time metrics after the original
    # implementation runs (it resets the counters to zero inside the log_interval branch).
    adv_before = total_loss_dict.get('advanced iterations', 0)
    skip_before = total_loss_dict.get('skipped iterations', 0)
    adv_after = adv_before + (0 if skipped_iter else 1)
    skip_after = skip_before + skipped_iter
    total_iterations_after = adv_after + skip_after if (adv_after + skip_after) > 0 else 1

    batch_size = args.micro_batch_size * args.data_parallel_size * get_num_microbatches()

    # Delegate to the original implementation to preserve baseline behaviour.
    report_memory_flag = _original_training_log(
        loss_dict,
        total_loss_dict,
        learning_rate,
        decoupled_learning_rate,
        iteration,
        loss_scale,
        report_memory_flag,
        skipped_iter,
        grad_norm,
        params_norm,
        num_zeros_in_grad,
    )

    wandb_writer = get_wandb_writer()
    if wandb_writer is None:
        return report_memory_flag

    log_metrics_interval = iteration % args.tensorboard_log_interval == 0
    logs: Dict[str, float] = {}

    if log_metrics_interval:
        logs['samples vs steps'] = float(args.consumed_train_samples)
        logs['train/global_step'] = float(iteration)
        logs['train/learning_rate'] = _maybe_item(learning_rate)
        logs['train/lr'] = _maybe_item(learning_rate)
        if decoupled_learning_rate is not None:
            logs['train/decoupled_lr'] = _maybe_item(decoupled_learning_rate)
        logs['train/batch_size'] = float(batch_size)
        logs['batch-size'] = float(batch_size)
        logs['train/consumed_samples'] = float(args.consumed_train_samples)
        logs['train/num_tokens'] = float(args.consumed_train_samples * args.seq_length)

        if args.train_samples:
            samples_per_epoch = float(args.train_samples)
            if getattr(args, 'epochs', 0):
                samples_per_epoch = max(1.0, float(args.train_samples) / float(args.epochs))
            if samples_per_epoch > 0:
                logs['train/epoch'] = float(args.consumed_train_samples) / samples_per_epoch

        # Map loss_dict entries into train/* namespace and capture a canonical train/loss if possible.
        for key, value in loss_dict.items():
            scalar = _maybe_item(value)
            sanitized = key.replace(' ', '_')
            logs[f'train/{sanitized}'] = scalar
            logs[key] = scalar
            if 'loss' in key.lower() and 'lm' in key.lower():
                logs.setdefault('train/loss', scalar)
                logs.setdefault('train/ppl', float(math.exp(min(20.0, scalar))))

        if args.skipped_train_samples > 0:
            logs['train/skipped_train_samples'] = float(args.skipped_train_samples)

        if args.log_loss_scale_to_tensorboard:
            logs['train/loss_scale'] = _maybe_item(loss_scale)

        logs['train/world_size'] = float(args.world_size)

        if grad_norm is not None:
            gn = _maybe_item(grad_norm)
            logs['train/grad_norm'] = gn
            logs['grad-norm'] = gn
        if num_zeros_in_grad is not None:
            nz = _maybe_item(num_zeros_in_grad)
            logs['train/num_zero_grads'] = nz
            logs['num-zeros'] = nz
        if params_norm is not None:
            pn = _maybe_item(params_norm)
            logs['train/params_norm'] = pn
            logs['params-norm'] = pn

        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            logs['train/mem_reserved_bytes'] = float(mem_stats['reserved_bytes.all.current'])
            logs['train/mem_allocated_bytes'] = float(mem_stats['allocated_bytes.all.current'])
            logs['train/mem_allocation_count'] = float(mem_stats['allocation.all.current'])

    # Attach the latest MoE auxiliary losses captured by the patched tracker.
    if _LAST_MOE_METRICS:
        for name, value in list(_LAST_MOE_METRICS.items()):
            logs[name] = float(value)
            sanitized = name.replace(' ', '_').replace('/', '_')
            logs[f'train/{sanitized}'] = float(value)
        if 'moe/aux_loss_total' in _LAST_MOE_METRICS:
            logs.setdefault('train/aux_loss', float(_LAST_MOE_METRICS['moe/aux_loss_total']))
        _LAST_MOE_METRICS.clear()

    # Emit timing/throughput metrics on the same cadence as Megatron's progress logging.
    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval-time').elapsed()
        elapsed_time_per_iteration = elapsed_time / max(1, total_iterations_after)
        iter_time_ms = elapsed_time_per_iteration * 1000.0
        logs['train/iteration_time_ms'] = iter_time_ms
        logs['time/step_ms'] = iter_time_ms

        flops_calc = (
            num_floating_point_operations_mla_moe
            if args.multi_latent_attention
            else num_floating_point_operations
        )
        throughput = flops_calc(args, batch_size) / (
            elapsed_time_per_iteration * 10**12 * args.world_size
        )
        logs['train/throughput_tflops_per_gpu'] = float(throughput)

    if logs:
        commit_flag = iteration % args.log_interval != 0
        wandb_writer.log(logs, step=iteration, commit=commit_flag)

    return report_memory_flag


def _apply_patch():
    # Avoid double patching in case the module gets imported multiple times.
    if getattr(training_module, '_wandb_logging_patched', False):
        return

    training_module.training_log = patched_training_log
    moe_utils_module.track_moe_metrics = patched_track_moe_metrics
    training_module._wandb_logging_patched = True


_apply_patch()

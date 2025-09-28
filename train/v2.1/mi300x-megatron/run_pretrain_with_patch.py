#!/usr/bin/env python3
"""Launch Megatron pretraining with custom W&B logging patches applied."""

import runpy  # noqa: E402  (imported after patch to avoid module execution)
import sys  # noqa: E402

# Importing these modules applies monkeypatches before Megatron boots.
import megatron_checkpoint_patch  # noqa: F401  (side effects only)
import megatron_wandb_logging  # noqa: F401  (side effects only)

if __name__ == "__main__":
    # Reuse the existing CLI by delegating to Megatron's pretrain entrypoint.
    runpy.run_module("pretrain_gpt", run_name="__main__")

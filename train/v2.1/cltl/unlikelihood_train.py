#!/usr/bin/env python3
"""
Unlikelihood Training for Language Leakage Correction
Optimized version using explicit leak/correction metadata

Usage:
    accelerate launch --num_processes 8 unlikelihood_train.py \
        --model_name_or_path your-org/qwen3-model \
        --dataset_name your-org/leak-corrections \
        --output_dir ./fixed-model
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from typing import List, Tuple, Optional, Dict
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _setup_file_logger(log_path: Optional[str]) -> Optional[logging.Handler]:
    """Attach a file handler so runs are captured to disk."""
    if not log_path:
        return None

    directory = os.path.dirname(log_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    handler = logging.FileHandler(log_path, mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return handler


# =============================================================================
# Dataset - Direct metadata version
# =============================================================================

class LeakCorrectionDataset(Dataset):
    """
    Dataset for unlikelihood training using explicit leak/correction metadata.
    
    Expected format:
    {
        "chosen": "...correct text with 結局...",
        "rejected": "...text with ultimately...",  # optional, used as fallback
        "leak_tokens": ["ultimately"],
        "corrected_token": "結局"
    }
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        max_length: int = 2048,
        chosen_col: str = "chosen",
        rejected_col: str = "rejected",
        leak_tokens_col: str = "leak_tokens",
        corrected_token_col: str = "corrected_token",
        meta_column: Optional[str] = "meta",
        meta_leak_tokens_key: Optional[str] = None,
        meta_corrected_token_key: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self.meta_column = meta_column or None
        self.meta_leak_tokens_key = meta_leak_tokens_key or leak_tokens_col
        self.meta_corrected_token_key = meta_corrected_token_key or corrected_token_col

        logger.info("Processing dataset with explicit leak/correction metadata...")

        skipped = 0
        multi_leak = 0

        for idx, row in enumerate(tqdm(hf_dataset, desc="Processing")):
            if not isinstance(row, dict):
                row = dict(row)
            chosen_text = row[chosen_col]
            leak_tokens = row.get(leak_tokens_col, [])
            corrected_token = row.get(corrected_token_col)

            if (not leak_tokens) and self.meta_column:
                leak_tokens = self._get_meta_value(row, self.meta_leak_tokens_key)
            if corrected_token in (None, "") and self.meta_column:
                corrected_token = self._get_meta_value(row, self.meta_corrected_token_key)

            if isinstance(leak_tokens, str):
                leak_tokens = [leak_tokens]
            leak_tokens = leak_tokens or []

            if isinstance(corrected_token, list):
                corrected_token = corrected_token[0] if corrected_token else None

            if not leak_tokens or not corrected_token:
                skipped += 1
                continue

            # Tokenize the chosen (correct) text
            chosen_ids = tokenizer.encode(
                chosen_text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length
            )

            # Get token IDs for leak and correction
            # Handle potential tokenization differences (spaces, etc.)
            bad_token_ids = []
            for leak_tok in leak_tokens:
                # Try with and without leading space
                candidates = [
                    tokenizer.encode(leak_tok, add_special_tokens=False),
                    tokenizer.encode(" " + leak_tok, add_special_tokens=False),
                    tokenizer.encode(leak_tok.strip(), add_special_tokens=False),
                ]
                for c in candidates:
                    if len(c) == 1:
                        bad_token_ids.append(c[0])
                        break
                else:
                    # Multi-token leak - take first token as primary target
                    if candidates[0]:
                        bad_token_ids.append(candidates[0][0])

            # Get good token ID
            good_candidates = [
                tokenizer.encode(corrected_token, add_special_tokens=False),
                tokenizer.encode(" " + corrected_token, add_special_tokens=False),
                tokenizer.encode(corrected_token.strip(), add_special_tokens=False),
            ]
            good_token_id = None
            for c in good_candidates:
                if len(c) == 1:
                    good_token_id = c[0]
                    break
            if good_token_id is None and good_candidates[0]:
                good_token_id = good_candidates[0][0]

            if not bad_token_ids:
                skipped += 1
                continue

            # Find position of good token in chosen sequence
            position = None
            for i, tid in enumerate(chosen_ids):
                if tid == good_token_id:
                    position = i
                    break

            if position is None or position == 0:
                # Fallback: diff with rejected if available
                if rejected_col in row and row[rejected_col]:
                    rejected_ids = tokenizer.encode(
                        row[rejected_col],
                        add_special_tokens=True,
                        truncation=True,
                        max_length=max_length
                    )
                    for i in range(1, min(len(chosen_ids), len(rejected_ids))):
                        if chosen_ids[i] != rejected_ids[i]:
                            position = i
                            break

            if position is None or position == 0:
                skipped += 1
                continue

            if len(leak_tokens) > 1:
                multi_leak += 1

            self.examples.append({
                "input_ids": chosen_ids,
                "bad_token_ids": bad_token_ids,  # List - can have multiple leak variants
                "good_token_id": good_token_id,
                "position": position,
                # Debug info
                "_leak_tokens": leak_tokens,
                "_corrected_token": corrected_token,
            })

        logger.info(f"Created {len(self.examples)} training examples")
        logger.info(f"Skipped {skipped} rows (missing metadata or position)")
        if multi_leak > 0:
            logger.info(f"Note: {multi_leak} examples have multiple leak tokens")

        # Log samples
        self._log_samples()

    def _get_meta_value(self, row: dict, key: str):
        if not self.meta_column or self.meta_column not in row:
            return None

        meta_value = row[self.meta_column]
        if meta_value is None:
            return None

        if isinstance(meta_value, str):
            meta_str = meta_value.strip()
            if not meta_str:
                return None
            try:
                meta_value = json.loads(meta_str)
            except json.JSONDecodeError:
                return None

        if isinstance(meta_value, dict):
            return meta_value.get(key)

        return None

    def _log_samples(self, num_samples: int = 5):
        if not self.examples:
            return

        logger.info("Sample corrections:")
        for ex in self.examples[:num_samples]:
            bad_strs = [self.tokenizer.decode([tid]) for tid in ex["bad_token_ids"]]
            good_str = self.tokenizer.decode([ex["good_token_id"]])
            logger.info(f"  Position {ex['position']}: {bad_strs} (bad) -> '{good_str}' (good)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, pad_token_id: int, max_bad_tokens: int = 4):
    """
    Collate with support for multiple bad tokens per example.
    """
    max_len = max(len(ex["input_ids"]) for ex in batch)
    batch_size = len(batch)

    input_ids = []
    attention_mask = []
    positions = []
    good_token_ids = []
    
    # Support multiple bad tokens per example
    all_bad_token_ids = []
    all_bad_masks = []

    for ex in batch:
        seq_len = len(ex["input_ids"])
        padding_len = max_len - seq_len

        input_ids.append(ex["input_ids"] + [pad_token_id] * padding_len)
        attention_mask.append([1] * seq_len + [0] * padding_len)
        positions.append(ex["position"])
        good_token_ids.append(ex["good_token_id"] if ex["good_token_id"] else -100)

        # Pad bad tokens
        bad_toks = ex["bad_token_ids"][:max_bad_tokens]
        bad_mask = [1.0] * len(bad_toks)
        pad_count = max_bad_tokens - len(bad_toks)
        bad_toks = bad_toks + [0] * pad_count
        bad_mask = bad_mask + [0.0] * pad_count
        
        all_bad_token_ids.append(bad_toks)
        all_bad_masks.append(bad_mask)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "positions": torch.tensor(positions, dtype=torch.long),
        "good_token_ids": torch.tensor(good_token_ids, dtype=torch.long),
        "bad_token_ids": torch.tensor(all_bad_token_ids, dtype=torch.long),  # (batch, max_bad)
        "bad_mask": torch.tensor(all_bad_masks, dtype=torch.float),
    }


# =============================================================================
# Loss Functions
# =============================================================================

def unlikelihood_loss(
    logits: torch.Tensor,           # (batch, seq_len, vocab)
    bad_token_ids: torch.Tensor,    # (batch, num_bad) - multiple bad tokens
    bad_mask: torch.Tensor,         # (batch, num_bad) - which are valid
    positions: torch.Tensor,        # (batch,)
) -> torch.Tensor:
    """
    Unlikelihood loss supporting multiple bad tokens per position.
    Loss = sum over bad tokens of -log(1 - p(bad_token | context))
    """
    batch_size, num_bad = bad_token_ids.shape
    device = logits.device
    batch_indices = torch.arange(batch_size, device=device)

    # Get logits at prediction position (pos-1 predicts token at pos)
    pred_positions = (positions - 1).clamp(min=0)
    position_logits = logits[batch_indices, pred_positions]  # (batch, vocab)
    
    probs = F.softmax(position_logits, dim=-1)  # (batch, vocab)

    # Gather probs for all bad tokens
    # Expand for gathering: (batch, num_bad)
    bad_probs = probs.gather(1, bad_token_ids)  # (batch, num_bad)

    # Unlikelihood: -log(1 - p) = -log1p(-p)
    bad_probs_clamped = bad_probs.clamp(max=1 - 1e-7)
    loss_per_bad = -torch.log1p(-bad_probs_clamped)  # (batch, num_bad)

    # Mask and average
    masked_loss = (loss_per_bad * bad_mask).sum(dim=1)  # (batch,)
    num_valid = bad_mask.sum(dim=1).clamp(min=1)
    loss_per_example = masked_loss / num_valid

    return loss_per_example.mean()


def reinforcement_loss(
    logits: torch.Tensor,
    good_token_ids: torch.Tensor,   # (batch,)
    positions: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss to reinforce good tokens."""
    batch_size = logits.size(0)
    device = logits.device
    batch_indices = torch.arange(batch_size, device=device)

    valid_mask = good_token_ids != -100
    if not valid_mask.any():
        return torch.tensor(0.0, device=device)

    pred_positions = (positions - 1).clamp(min=0)
    position_logits = logits[batch_indices, pred_positions]

    ce_loss = F.cross_entropy(
        position_logits[valid_mask],
        good_token_ids[valid_mask],
        reduction='mean'
    )

    return ce_loss


def combined_loss(
    model,
    batch: Dict[str, torch.Tensor],
    alpha_ul: float = 1.0,
    alpha_reinforce: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """Combined unlikelihood + reinforcement loss."""

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"]
    )
    logits = outputs.logits

    ul_loss = unlikelihood_loss(
        logits,
        batch["bad_token_ids"],
        batch["bad_mask"],
        batch["positions"],
    )
    
    rf_loss = reinforcement_loss(
        logits,
        batch["good_token_ids"],
        batch["positions"],
    )

    total_loss = alpha_ul * ul_loss + alpha_reinforce * rf_loss

    metrics = {
        "ul_loss": ul_loss.item(),
        "rf_loss": rf_loss.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, metrics


# =============================================================================
# Diagnostics
# =============================================================================

@torch.no_grad()
def compute_token_probabilities(
    model,
    dataset: LeakCorrectionDataset,
    device: torch.device,
    num_samples: int = 100
) -> dict:
    """Compute average probabilities of bad/good tokens."""
    model.eval()

    bad_probs = []
    good_probs = []

    samples = min(num_samples, len(dataset))
    
    for i in range(samples):
        ex = dataset[i]
        input_ids = torch.tensor([ex["input_ids"]], device=device)

        outputs = model(input_ids)
        logits = outputs.logits[0]

        pos = ex["position"]
        pred_pos = max(0, pos - 1)
        probs = F.softmax(logits[pred_pos], dim=-1)

        # Average over all bad tokens
        for bad_tid in ex["bad_token_ids"]:
            bad_probs.append(probs[bad_tid].item())
        
        if ex["good_token_id"]:
            good_probs.append(probs[ex["good_token_id"]].item())

    model.train()

    return {
        "avg_bad_prob": sum(bad_probs) / len(bad_probs) if bad_probs else 0,
        "avg_good_prob": sum(good_probs) / len(good_probs) if good_probs else 0,
    }


@torch.no_grad()
def detailed_diagnostics(
    model,
    tokenizer,
    dataset: LeakCorrectionDataset,
    device: torch.device,
    num_samples: int = 10
):
    """Print detailed before/after diagnostics for specific examples."""
    model.eval()
    
    logger.info("\nDetailed token probability analysis:")
    logger.info("=" * 60)
    
    for i in range(min(num_samples, len(dataset))):
        ex = dataset[i]
        input_ids = torch.tensor([ex["input_ids"]], device=device)
        
        outputs = model(input_ids)
        logits = outputs.logits[0]
        
        pos = ex["position"]
        pred_pos = max(0, pos - 1)
        probs = F.softmax(logits[pred_pos], dim=-1)
        
        # Get context
        context = tokenizer.decode(ex["input_ids"][max(0, pred_pos-10):pred_pos+1])
        
        logger.info(f"\nExample {i+1}:")
        logger.info(f"  Context: ...{context}")
        
        for bad_tid in ex["bad_token_ids"]:
            bad_tok = tokenizer.decode([bad_tid])
            logger.info(f"  BAD  '{bad_tok}' (id={bad_tid}): {probs[bad_tid].item():.4%}")
        
        if ex["good_token_id"]:
            good_tok = tokenizer.decode([ex["good_token_id"]])
            logger.info(f"  GOOD '{good_tok}' (id={ex['good_token_id']}): {probs[ex['good_token_id']].item():.4%}")
        
        # Show top 5 predictions
        top5 = probs.topk(5)
        logger.info(f"  Top 5 predictions:")
        for prob, tid in zip(top5.values, top5.indices):
            tok = tokenizer.decode([tid])
            marker = ""
            if tid in ex["bad_token_ids"]:
                marker = " <-- BAD"
            elif tid == ex["good_token_id"]:
                marker = " <-- GOOD"
            logger.info(f"    '{tok}': {prob.item():.4%}{marker}")
    
    model.train()


# =============================================================================
# Training
# =============================================================================

def train(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    file_handler = None
    if accelerator.is_main_process:
        file_handler = _setup_file_logger(args.log_file)

    set_seed(args.seed)

    if accelerator.is_main_process:
        logger.info(f"Arguments: {args}")
        logger.info(f"Number of processes: {accelerator.num_processes}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.flash_attn else None,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    if args.dataset_name.endswith((".json", ".jsonl")):
        raw_dataset = load_dataset("json", data_files=args.dataset_name, split="train")
    else:
        raw_dataset = load_dataset(
            args.dataset_name,
            split=args.dataset_split,
            token=args.hf_token,
        )

    # Create dataset
    dataset = LeakCorrectionDataset(
        raw_dataset,
        tokenizer,
        max_length=args.max_length,
        chosen_col=args.chosen_column,
        rejected_col=args.rejected_column,
        leak_tokens_col=args.leak_tokens_column,
        corrected_token_col=args.corrected_token_column,
        meta_column=args.meta_column,
        meta_leak_tokens_key=args.meta_leak_tokens_key,
        meta_corrected_token_key=args.meta_corrected_token_key,
    )

    if len(dataset) == 0:
        raise ValueError("No training examples found!")

    # Dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    num_update_steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    total_steps = num_update_steps_per_epoch * args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    # Prepare
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # Pre-training diagnostics
    if accelerator.is_main_process:
        logger.info("Pre-training diagnostics...")
        pre_probs = compute_token_probabilities(
            accelerator.unwrap_model(model), 
            dataset, 
            accelerator.device
        )
        logger.info(f"Pre-training - Avg bad token prob: {pre_probs['avg_bad_prob']:.4%}")
        logger.info(f"Pre-training - Avg good token prob: {pre_probs['avg_good_prob']:.4%}")
        
        if args.verbose_diagnostics:
            detailed_diagnostics(
                accelerator.unwrap_model(model),
                tokenizer,
                dataset,
                accelerator.device
            )

    # Training loop
    logger.info("Starting training...")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_ul_loss = 0
        epoch_rf_loss = 0
        num_batches = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=not accelerator.is_main_process,
        )

        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                loss, metrics = combined_loss(
                    model,
                    batch,
                    alpha_ul=args.alpha_ul,
                    alpha_reinforce=args.alpha_reinforce,
                )

                accelerator.backward(loss)

                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += metrics["total_loss"]
            epoch_ul_loss += metrics["ul_loss"]
            epoch_rf_loss += metrics["rf_loss"]
            num_batches += 1

            if accelerator.is_main_process:
                progress_bar.set_postfix({
                    "loss": f"{metrics['total_loss']:.4f}",
                    "ul": f"{metrics['ul_loss']:.4f}",
                    "rf": f"{metrics['rf_loss']:.4f}",
                })

            global_step += 1

        if accelerator.is_main_process:
            logger.info(
                f"Epoch {epoch + 1} - "
                f"Loss: {epoch_loss / num_batches:.4f}, "
                f"UL: {epoch_ul_loss / num_batches:.4f}, "
                f"RF: {epoch_rf_loss / num_batches:.4f}"
            )

    # Post-training diagnostics
    if accelerator.is_main_process:
        logger.info("Post-training diagnostics...")
        post_probs = compute_token_probabilities(
            accelerator.unwrap_model(model),
            dataset,
            accelerator.device
        )
        logger.info(f"Post-training - Avg bad token prob: {post_probs['avg_bad_prob']:.4%}")
        logger.info(f"Post-training - Avg good token prob: {post_probs['avg_good_prob']:.4%}")
        logger.info(
            f"Change - Bad: {post_probs['avg_bad_prob'] - pre_probs['avg_bad_prob']:+.4%}, "
            f"Good: {post_probs['avg_good_prob'] - pre_probs['avg_good_prob']:+.4%}"
        )
        
        if args.verbose_diagnostics:
            detailed_diagnostics(
                accelerator.unwrap_model(model),
                tokenizer,
                dataset,
                accelerator.device
            )

    # Save
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info(f"Saving model to {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        args.output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        safe_serialization=True,
    )

    if accelerator.is_main_process:
        tokenizer.save_pretrained(args.output_dir)
        logger.info("Done!")

        if file_handler:
            logger.removeHandler(file_handler)


def parse_args():
    parser = argparse.ArgumentParser()

    # Model/data
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hf_token", type=str, default=None)

    # Column names
    parser.add_argument("--chosen_column", type=str, default="chosen")
    parser.add_argument("--rejected_column", type=str, default="rejected")
    parser.add_argument("--leak_tokens_column", type=str, default="leak_tokens")
    parser.add_argument("--corrected_token_column", type=str, default="corrected_token")
    parser.add_argument("--meta_column", type=str, default="meta",
                        help="Optional column that stores leak metadata as a dict or JSON string")
    parser.add_argument("--meta_leak_tokens_key", type=str, default=None,
                        help="Key inside meta column for leak tokens (defaults to leak_tokens_column)")
    parser.add_argument("--meta_corrected_token_key", type=str, default=None,
                        help="Key inside meta column for corrected token")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Optional path to append training logs for run tracking")

    # Training
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=2048)

    # Loss weights
    parser.add_argument("--alpha_ul", type=float, default=1.0)
    parser.add_argument("--alpha_reinforce", type=float, default=0.1)

    # Performance
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--flash_attn", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)

    # Diagnostics
    parser.add_argument("--verbose_diagnostics", action="store_true",
                        help="Print detailed per-example token probabilities")

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

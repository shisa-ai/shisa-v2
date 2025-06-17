#!/usr/bin/env python3
"""
Detect Axolotl “last‑turn not trainable” cases in HF chat datasets
-----------------------------------------------------------------

Example
-------
>>> DATASETS = [
...     "HuggingFaceH4/ultrachat_200k",
...     "shisa-ai/shisa-v2-roleplaying-sft",
... ]
>>> summary, problems = scan_datasets(DATASETS)
>>> print(summary)
>>> problems.to_parquet("bad_samples.parquet")

Requires: datasets, pandas, tqdm (optional but nice).
"""
from __future__ import annotations

import os
from typing import Iterable, List, Tuple

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm


# ------------------------------------------------------------
# Core helpers
# ------------------------------------------------------------
DEFAULT_ROLES_TO_TRAIN: Tuple[str, ...] = ("assistant",)
POSSIBLE_MESSAGE_KEYS: Tuple[str, ...] = ("messages", "conversations", "conversation")


def _infer_message_key(example: dict) -> str | None:
    """Best‑effort guess of which column holds the chat turns."""
    for key in POSSIBLE_MESSAGE_KEYS:
        if key in example and isinstance(example[key], list):
            return key
    return None


def _last_turn_not_trainable(messages: List[dict], roles_to_train: Tuple[str, ...]) -> bool:
    """
    Return True if **the very last message** is *not* from a role we train on.

    The function is intentionally simple – if you use more complex
    per‑turn training logic (e.g. `trainable: bool` keys) extend here.
    """
    if not messages:
        return False  # empty example: ignore
    last_role = messages[-1].get("role")
    return last_role not in roles_to_train


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def scan_datasets(
    dataset_names: Iterable[str],
    split: str = "train",
    roles_to_train: Tuple[str, ...] = DEFAULT_ROLES_TO_TRAIN,
    streaming: bool = False,
    hf_token: str | None = None,
):
    """
    Iterate through `dataset_names`, identify records whose final turn is *not*
    trainable, and return `(summary_df, problem_df)`.

    Parameters
    ----------
    dataset_names: iterable of str
        Hugging Face dataset identifiers (`repo` or `namespace/repo`).
    split: str
        Which split to scan (default `"train"`).
    roles_to_train: tuple of str
        The set of roles considered trainable (default `("assistant",)`).
    streaming: bool
        Pass `streaming=True` if datasets are > RAM.
    hf_token: str, optional
        Hugging Face auth token for private datasets.
    """
    summary_rows = []
    problem_rows = []

    for dname in dataset_names:
        print(f"→ scanning {dname!r} ({split} split)…")
        ds = load_dataset(dname, split=split, streaming=streaming, token=hf_token)

        total = 0
        bad = 0

        iterator = ds if streaming else tqdm(ds, unit=" rows")

        for idx, ex in enumerate(iterator):
            total += 1
            msg_key = _infer_message_key(ex)
            if msg_key is None:
                continue  # not a chat example; skip
            if _last_turn_not_trainable(ex[msg_key], roles_to_train):
                bad += 1
                problem_rows.append(
                    {
                        "dataset": dname,
                        "row_id": idx,
                        "last_role": ex[msg_key][-1].get("role"),
                        "num_turns": len(ex[msg_key]),
                    }
                )

        summary_rows.append(
            {
                "dataset": dname,
                "total_rows": total,
                "bad_rows": bad,
                "percent_bad": round(100 * bad / total, 3) if total else 0.0,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("percent_bad", ascending=False)
    problem_df = pd.DataFrame(problem_rows)

    return summary_df, problem_df


# ------------------------------------------------------------
# CLI entry‑point
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "datasets",
        nargs="+",
        help="List/JSON‑array of dataset names, or path to a .txt/.json file containing them.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--roles", default="assistant", help="Comma‑separated list.")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--out_summary", default="summary.csv")
    parser.add_argument("--out_problems", default="problem_rows.parquet")
    args = parser.parse_args()

    # Resolve dataset list argument
    if len(args.datasets) == 1 and args.datasets[0].endswith((".txt", ".json")):
        path = args.datasets[0]
        if path.endswith(".txt"):
            with open(path) as fh:
                dsets = [ln.strip() for ln in fh if ln.strip()]
        else:
            dsets = json.loads(open(path).read())
    else:
        dsets = args.datasets

    summary, problems = scan_datasets(
        dsets,
        split=args.split,
        roles_to_train=tuple(r.strip() for r in args.roles.split(",")),
        streaming=args.streaming,
        hf_token=args.hf_token,
    )

    summary.to_csv(args.out_summary, index=False)
    problems.to_parquet(args.out_problems)

    print(f"\nSummary written to {args.out_summary}; "
          f"problem rows to {args.out_problems}.")


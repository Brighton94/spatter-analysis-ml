"""SegFormer trainer powered by Hugging-Face Trainer."""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from transformers import EvalPrediction, Trainer, TrainingArguments

from datasets.multi_build import build_dataset_from_keys
from models.segformer_baseline import load_model


#  CLI
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--keys",
        nargs="+",
        required=True,
        help="Dataset keys defined in config.DATASET_PATHS",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--accum_steps", type=int, default=2)
    return ap.parse_args()


#  Metrics


def mean_iou_from_logits(
    logits: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5
) -> float:
    """Compute mean IoU from logits and masks."""

    preds = (torch.sigmoid(logits) > threshold).bool()
    inter = torch.logical_and(preds, masks.bool()).sum()
    union = torch.logical_or(preds, masks.bool()).sum()
    return (inter / (union + 1e-6)).item()


def compute_metrics(p: EvalPrediction) -> dict[str, float]:
    """Hugging-Face callback: receives numpy arrays."""
    logits, labels = p.predictions, p.label_ids
    logits = torch.from_numpy(logits).squeeze(1)  # [B, H, W]
    labels = torch.from_numpy(labels)
    miou = mean_iou_from_logits(logits, labels)
    bce = F.binary_cross_entropy_with_logits(logits.float(), labels.float()).item()
    return {"mean_iou": miou, "bce": bce}


#  Collator


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """The dataset returns (image, mask) tuples.

    Trainer expects a dict with keys matching the model forward arg names.
    """
    imgs = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    return {"pixel_values": imgs, "labels": masks}


def main() -> None:
    args = parse_args()

    # dataset
    full_ds = build_dataset_from_keys(args.keys, size=512, augment=True)
    n_val = int(len(full_ds) * args.val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    # model
    _, model = load_model()
    torch.backends.cuda.matmul.allow_tf32 = True
    model.gradient_checkpointing_enable()
    model = torch.compile(model)

    # HF Trainer
    training_args = TrainingArguments(
        output_dir="runs/segformer",
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        fp16=True,
        optim="adamw_torch_fused",
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        tf32=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        gradient_accumulation_steps=args.accum_steps,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=args.workers,
        dataloader_pin_memory=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="mean_iou",
        greater_is_better=True,
        report_to=["mlflow"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()

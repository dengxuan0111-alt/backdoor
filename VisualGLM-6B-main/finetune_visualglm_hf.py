#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import argparse
from typing import List, Dict, Any, Optional

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoTokenizer, AutoModel, get_scheduler
from peft import LoraConfig, TaskType, get_peft_model


DEFAULT_MODEL_DIR = "/home/dengxuan/VisualGLM-6B-main"


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()


def reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if not is_dist_avail_and_initialized():
        return t
    rt = t.detach().clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


def print_rank0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1 and not is_dist_avail_and_initialized():
        deepspeed.init_distributed()

    return local_rank, world_size


class BlipImageEvalProcessor:
    def __init__(self, image_size=224):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __call__(self, image: Image.Image):
        return self.transform(image)


class FewShotDataset(Dataset):
    def __init__(self, path, processor, tokenizer, image_length, args):
        max_seq_length = args.max_source_length + args.max_target_length

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.samples = []
        for item in data:
            image = processor(Image.open(item["img"]).convert("RGB"))

            input0 = tokenizer.encode("<img>", add_special_tokens=False)
            input1 = [tokenizer.unk_token_id] * image_length
            input2 = tokenizer.encode(
                "</img>问：" + item["prompt"] + "\n答：",
                add_special_tokens=False
            )

            a_ids = input0 + input1 + input2
            b_ids = tokenizer.encode(item["label"], add_special_tokens=False)

            if len(a_ids) > args.max_source_length - 1:
                a_ids = a_ids[: args.max_source_length - 1]
            if len(b_ids) > args.max_target_length - 2:
                b_ids = b_ids[: args.max_target_length - 2]

            pre_image_length = len(input0)
            input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

            context_length = input_ids.index(tokenizer.bos_token_id)
            mask_position = context_length - 1
            labels = [-100] * context_length + input_ids[mask_position + 1:]

            pad_len = max_seq_length - len(input_ids)
            if pad_len < 0:
                input_ids = input_ids[:max_seq_length]
                labels = labels[:max_seq_length]
            else:
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len

            if args.ignore_pad_token_for_loss:
                labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

            self.samples.append(
                {
                    "image": image,
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "pre_image_length": pre_image_length,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def data_collator(examples: List[Dict[str, Any]]):
    return {
        "input_ids": torch.stack([ex["input_ids"] for ex in examples]),
        "labels": torch.stack([ex["labels"] for ex in examples]),
        "images": torch.stack([ex["image"] for ex in examples]),
        "pre_image_length": examples[0]["pre_image_length"],
    }


def apply_lora_layer_range_if_needed(model, layer_range: Optional[List[int]]):
    if not layer_range:
        return

    if len(layer_range) != 2:
        raise ValueError("--layer_range 需要两个整数，例如: --layer_range 0 14")

    start, end = layer_range
    layer_pat = re.compile(r"\.layers\.(\d+)\.")

    kept = 0
    frozen = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" not in name:
            continue

        m = layer_pat.search(name)
        if m is None:
            kept += 1
            continue

        layer_id = int(m.group(1))
        if start <= layer_id < end:
            kept += 1
        else:
            param.requires_grad_(False)
            frozen += 1

    print_rank0(f"[LoRA] layer_range applied: keep [{start}, {end}), kept={kept}, frozen={frozen}")


def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, p in model.named_parameters():
        n = p.numel()
        all_params += n
        if p.requires_grad:
            trainable_params += n
    pct = 100.0 * trainable_params / all_params if all_params > 0 else 0.0
    print_rank0(
        f"trainable params: {trainable_params:,d} | "
        f"all params: {all_params:,d} | "
        f"trainable%: {pct:.6f}"
    )


def build_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    if args.use_ptuning:
        raise NotImplementedError("当前脚本暂未实现 --use_ptuning")
    if args.use_qlora:
        raise NotImplementedError("当前脚本暂未实现 --use_qlora")

    torch_dtype = torch.float16 if args.fp16 else torch.float32

    model = AutoModel.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch_dtype,
    )

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if args.use_lora:
        target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        apply_lora_layer_range_if_needed(model, args.layer_range)

    print_trainable_parameters(model)
    return model, tokenizer


def save_hf_artifacts(model_engine, tokenizer, output_dir, tag):
    if not is_main_process():
        return

    save_dir = os.path.join(output_dir, tag)
    os.makedirs(save_dir, exist_ok=True)

    module = model_engine.module
    module.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print_rank0(f"[save] HF artifacts saved to: {save_dir}")


def save_deepspeed_checkpoint(model_engine, output_dir, tag):
    ds_ckpt_dir = os.path.join(output_dir, "deepspeed_ckpt")
    os.makedirs(ds_ckpt_dir, exist_ok=True)
    model_engine.save_checkpoint(ds_ckpt_dir, tag=tag)
    if is_main_process():
        print(f"[save] DeepSpeed checkpoint saved to: {ds_ckpt_dir} (tag={tag})")


def build_optimizer_and_scheduler(args, model, num_training_steps):
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    warmup_steps = args.warmup_steps if args.warmup_steps >= 0 else int(args.warmup_ratio * num_training_steps)

    scheduler_name = args.lr_decay_style.lower()
    if scheduler_name == "cosine":
        hf_scheduler_name = "cosine"
    elif scheduler_name in ("linear", "constant", "constant_with_warmup"):
        hf_scheduler_name = scheduler_name
    else:
        print_rank0(f"[warn] 不支持 lr_decay_style={args.lr_decay_style}，改用 cosine")
        hf_scheduler_name = "cosine"

    scheduler = get_scheduler(
        name=hf_scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def build_deepspeed_config(args, world_size):
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "train_batch_size": args.batch_size * args.gradient_accumulation_steps * world_size,
        "steps_per_print": args.log_interval,
        "gradient_clipping": args.max_grad_norm,
        "zero_optimization": {
            "stage": args.zero_stage,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
    }

    if args.fp16:
        ds_config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 12,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1,
        }

    return ds_config


def train(args):
    local_rank, world_size = setup_distributed()

    model, tokenizer = build_model_and_tokenizer(args)
    image_processor = BlipImageEvalProcessor(224)

    image_length = model.config.image_length
    train_dataset = FewShotDataset(
        args.train_data,
        image_processor,
        tokenizer,
        image_length,
        args,
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=get_rank(),
        shuffle=True,
        drop_last=False,
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=data_collator,
        drop_last=False,
    )

    updates_per_epoch = max(1, math.ceil(len(train_loader) / args.gradient_accumulation_steps))

    if args.train_iters is not None and args.train_iters > 0:
        max_train_steps = args.train_iters
        num_epochs = math.ceil(max_train_steps / updates_per_epoch)
    else:
        max_train_steps = args.epochs * updates_per_epoch
        num_epochs = args.epochs

    optimizer, scheduler = build_optimizer_and_scheduler(args, model, max_train_steps)
    ds_config = build_deepspeed_config(args, world_size)

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config,
    )

    device = model_engine.device
    image_dtype = torch.float16 if args.fp16 else torch.float32

    print_rank0(f"device={device}, world_size={world_size}, local_rank={local_rank}")
    print_rank0(f"dataset_size={len(train_dataset)}, updates_per_epoch={updates_per_epoch}, max_train_steps={max_train_steps}")

    global_step = 0
    completed_epochs = 0

    for epoch in range(num_epochs):
        completed_epochs += 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model_engine.train()
        running_loss = 0.0
        step_in_epoch = 0

        pbar = tqdm(
            train_loader,
            disable=not is_main_process(),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
        )

        for batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            images = batch["images"].to(device=device, dtype=image_dtype, non_blocking=True)
            pre_image_length = batch["pre_image_length"]

            outputs = model_engine(
                input_ids=input_ids,
                labels=labels,
                images=images,
                pre_image_length=pre_image_length,
                return_dict=True,
            )

            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            model_engine.backward(loss)
            model_engine.step()

            loss_mean = reduce_mean(loss.detach()).item()
            running_loss += loss_mean
            step_in_epoch += 1

            if model_engine.is_gradient_accumulation_boundary():
                global_step += 1

                if is_main_process():
                    current_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.lr
                    pbar.set_postfix(
                        loss=f"{loss_mean:.4f}",
                        lr=f"{current_lr:.3e}",
                        gstep=global_step,
                    )

                if args.save_interval > 0 and global_step % args.save_interval == 0:
                    tag = f"global_step{global_step}"
                    save_deepspeed_checkpoint(model_engine, args.output_dir, tag)
                    barrier()
                    save_hf_artifacts(model_engine, tokenizer, args.output_dir, tag)
                    barrier()

                if global_step >= max_train_steps:
                    break

        avg_epoch_loss = running_loss / max(step_in_epoch, 1)
        print_rank0(f"[epoch {epoch + 1}] avg_loss={avg_epoch_loss:.6f}, global_step={global_step}")

        if args.save_epoch:
            tag = f"epoch-{epoch + 1}"
            save_deepspeed_checkpoint(model_engine, args.output_dir, tag)
            barrier()
            save_hf_artifacts(model_engine, tokenizer, args.output_dir, tag)
            barrier()

        if global_step >= max_train_steps:
            break

    final_tag = "final"
    save_deepspeed_checkpoint(model_engine, args.output_dir, final_tag)
    barrier()
    save_hf_artifacts(model_engine, tokenizer, args.output_dir, final_tag)
    barrier()

    print_rank0(
        f"Training finished. completed_epochs={completed_epochs}, "
        f"global_step={global_step}, output_dir={args.output_dir}"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))

    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--train_data", "--train-data", dest="train_data", type=str, required=True)
    parser.add_argument("--valid_data", "--valid-data", dest="valid_data", type=str, default=None)
    parser.add_argument("--output_dir", "--save", dest="output_dir", type=str, default="./checkpoints_hf_ds")

    parser.add_argument("--max_source_length", "--max-source-length", dest="max_source_length", type=int, default=64)
    parser.add_argument("--max_target_length", "--max-target-length", dest="max_target_length", type=int, default=256)
    parser.add_argument("--ignore_pad_token_for_loss", type=bool, default=True)

    parser.add_argument("--train_iters", "--train-iters", dest="train_iters", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", "--eval-batch-size", dest="eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", "--warmup", dest="warmup_ratio", type=float, default=0.02)
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--lr_decay_style", "--lr-decay-style", dest="lr_decay_style", type=str, default="cosine")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--zero_stage", "--zero-stage", dest="zero_stage", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--log_interval", type=int, default=10)

    parser.add_argument("--save_interval", "--save-interval", dest="save_interval", type=int, default=300)
    parser.add_argument("--save_epoch", action="store_true")

    parser.add_argument("--checkpoint_activations", "--checkpoint-activations", dest="gradient_checkpointing", action="store_true")

    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--use_ptuning", action="store_true")
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--pre_seq_len", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=10)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", type=str, default="query_key_value")
    parser.add_argument("--layer_range", nargs="+", type=int, default=None)

    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--model-parallel-size", type=int, default=1)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--resume-dataloader", action="store_true")
    parser.add_argument("--distributed-backend", type=str, default="nccl")
    parser.add_argument("--eval_interval", "--eval-interval", dest="eval_interval", type=int, default=10000)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--eval-iters", dest="eval_iters", type=int, default=10)
    parser.add_argument("--skip-init", action="store_true")

    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"model_dir 不存在: {args.model_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
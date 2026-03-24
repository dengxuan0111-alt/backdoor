from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--save-path", type=str, default="data/sample_batch.pt")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    images, labels = next(iter(loader))
    torch.save(
        {
            "images": images,
            "labels": labels,
        },
        args.save_path,
    )
    print(f"saved to {args.save_path}, images={tuple(images.shape)}, labels={tuple(labels.shape)}")


if __name__ == "__main__":
    main()
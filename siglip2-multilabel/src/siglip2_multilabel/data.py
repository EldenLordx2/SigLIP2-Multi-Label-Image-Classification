from dataclasses import dataclass
from typing import Any, List

import torch
from PIL import Image
from torch.utils.data import Dataset


class MultiLabelImageDataset(Dataset):
    def __init__(self, txt_path: str, num_labels: int, transform=None, dummy_size=(448, 448)):
        self.samples = []
        self.num_labels = num_labels
        self.bad_image_count = 0
        self.dummy_size = dummy_size
        self.transform = transform

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    continue
                img_path, label_str = parts
                label_list = [int(x) for x in label_str.split(",") if x.strip() != ""]
                if len(label_list) != num_labels:
                    continue
                self.samples.append((img_path, label_list))

        print(f"Loaded {len(self.samples)} samples from {txt_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_list = self.samples[idx]
        labels = torch.tensor(label_list, dtype=torch.float32)

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            valid = torch.tensor(1.0, dtype=torch.float32)
        except Exception:
            self.bad_image_count += 1
            image = Image.new("RGB", self.dummy_size, color=(0, 0, 0))
            if self.transform is not None:
                image = self.transform(image)
            labels = torch.zeros_like(labels)
            valid = torch.tensor(0.0, dtype=torch.float32)

        return {"image": image, "labels": labels, "valid": valid}


class ImageOnlyDataset(Dataset):
    def __init__(self, txt_path: str, dummy_size=(256, 256)):
        self.paths: List[str] = []
        self.dummy_size = dummy_size
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if p:
                    self.paths.append(p)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        read_error = False
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", self.dummy_size, color=(0, 0, 0))
            read_error = True
        return {"image": image, "path": img_path, "read_error": read_error}


@dataclass
class Collator:
    processor: Any
    disable_resize_ops: bool = True

    def __call__(self, batch):
        images = [b["image"] for b in batch]
        try:
            if self.disable_resize_ops:
                inputs = self.processor(
                    images=images,
                    return_tensors="pt",
                    do_resize=False,
                    do_rescale=True,
                    do_normalize=True,
                )
            else:
                inputs = self.processor(images=images, return_tensors="pt")
        except TypeError:
            inputs = self.processor(images=images, return_tensors="pt")

        result = {"pixel_values": inputs["pixel_values"]}
        if "labels" in batch[0]:
            result["labels"] = torch.stack([b["labels"] for b in batch])
            result["valid"] = torch.stack([b["valid"] for b in batch])
        else:
            result["paths"] = [b["path"] for b in batch]
            result["read_errors"] = [b["read_error"] for b in batch]
        return result

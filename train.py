import argparse
import json
import os
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import open_clip


@dataclass
class TrainConfig:
    dataset_name: str = "nlphuji/flickr30k"
    dataset_split: str = "test"
    train_size: int = 30000
    eval_size: int = 1000
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-5
    weight_decay: float = 0.05
    temperature: float = 0.07
    kd_alpha: float = 0.5
    use_kd: bool = False
    fewshot_aug: bool = False
    output_dir: str = "experiments_3080/results/full"
    seed: int = 42
    num_workers: int = 4


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_fixed_splits(dataset_len: int, seed: int, eval_size: int, train_size: int):
    """Create deterministic non-overlapping eval/train indices.

    Same logic is used by eval.py:
    - shuffle all indices by seed
    - first eval_size indices are held out for evaluation
    - train data is drawn only from the remaining pool
    """
    if eval_size <= 0:
        raise ValueError("eval_size must be > 0")
    if eval_size >= dataset_len:
        raise ValueError(f"eval_size={eval_size} must be smaller than dataset size={dataset_len}")

    rng = random.Random(seed)
    all_indices = list(range(dataset_len))
    rng.shuffle(all_indices)

    eval_indices = all_indices[:eval_size]
    train_pool = all_indices[eval_size:]
    train_indices = train_pool[: min(train_size, len(train_pool))]
    return train_indices, eval_indices


class Flickr30KDataset(Dataset):
    def __init__(self, hf_dataset, preprocess, fewshot_aug=False):
        self.ds = hf_dataset
        self.preprocess = preprocess
        self.fewshot_aug = fewshot_aug

    def __len__(self):
        return len(self.ds)

    def _get_caption(self, item):
        captions = item.get("caption", None)
        if captions is None:
            captions = item.get("captions", None)

        if isinstance(captions, list):
            caption = random.choice(captions)
        else:
            caption = str(captions)

        if self.fewshot_aug:
            templates = [
                "{}",
                "a photo of {}",
                "an image showing {}",
                "a detailed photo of {}",
            ]
            caption = random.choice(templates).format(caption)

        return caption

    def __getitem__(self, idx):
        item = self.ds[idx]

        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        caption = self._get_caption(item)
        image = self.preprocess(image)
        return image, caption


def contrastive_loss(image_features, text_features, temperature=0.07):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    logits = image_features @ text_features.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


def kd_loss(student_img, student_txt, teacher_img, teacher_txt, temperature=0.07):
    student_img = F.normalize(student_img, dim=-1)
    student_txt = F.normalize(student_txt, dim=-1)
    teacher_img = F.normalize(teacher_img, dim=-1)
    teacher_txt = F.normalize(teacher_txt, dim=-1)

    s_logits = student_img @ student_txt.T / temperature
    t_logits = teacher_img @ teacher_txt.T / temperature

    return F.kl_div(
        F.log_softmax(s_logits, dim=-1),
        F.softmax(t_logits, dim=-1),
        reduction="batchmean",
    )


def freeze_teacher(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


def save_split_info(output_dir, cfg, train_indices, eval_indices, dataset_len):
    path = os.path.join(output_dir, "split_info.json")
    info = {
        "dataset_name": cfg.dataset_name,
        "dataset_split": cfg.dataset_split,
        "dataset_len": dataset_len,
        "seed": cfg.seed,
        "train_size": len(train_indices),
        "eval_size": len(eval_indices),
        "train_eval_overlap": len(set(train_indices) & set(eval_indices)),
        "train_indices_preview": train_indices[:20],
        "eval_indices_preview": eval_indices[:20],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def train(cfg: TrainConfig):
    seed_everything(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    os.makedirs(cfg.output_dir, exist_ok=True)

    student, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained="openai",
        device=device,
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    teacher = None
    if cfg.use_kd:
        teacher, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16",
            pretrained="laion2b_s34b_b88k",
            device=device,
        )
        freeze_teacher(teacher)

    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    train_indices, eval_indices = build_fixed_splits(
        dataset_len=len(ds),
        seed=cfg.seed,
        eval_size=cfg.eval_size,
        train_size=cfg.train_size,
    )

    overlap = len(set(train_indices) & set(eval_indices))
    print(f"Dataset: {cfg.dataset_name} / split={cfg.dataset_split}")
    print(f"Dataset size: {len(ds)}")
    print(f"Eval holdout size: {len(eval_indices)}")
    print(f"Train size: {len(train_indices)}")
    print(f"Train/eval overlap: {overlap}")
    print(f"LR: {cfg.lr} | Epochs: {cfg.epochs} | KD: {cfg.use_kd} | Few-shot aug: {cfg.fewshot_aug}")
    if overlap != 0:
        raise RuntimeError("Train/eval overlap is not zero. Stop to avoid data leakage.")

    save_split_info(cfg.output_dir, cfg, train_indices, eval_indices, len(ds))

    train_ds = Flickr30KDataset(
        Subset(ds, train_indices),
        preprocess=preprocess,
        fewshot_aug=cfg.fewshot_aug,
    )

    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=use_amp,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    student.train()

    for epoch in range(cfg.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")
        total_loss = 0.0

        for images, captions in pbar:
            images = images.to(device, non_blocking=True)
            texts = tokenizer(list(captions)).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                s_img = student.encode_image(images)
                s_txt = student.encode_text(texts)

                ce_loss = contrastive_loss(s_img, s_txt, temperature=cfg.temperature)

                if cfg.use_kd:
                    with torch.no_grad():
                        t_img = teacher.encode_image(images)
                        t_txt = teacher.encode_text(texts)

                    distill = kd_loss(s_img, s_txt, t_img, t_txt, temperature=cfg.temperature)
                    loss = (1 - cfg.kd_alpha) * ce_loss + cfg.kd_alpha * distill
                else:
                    loss = ce_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}")

        ckpt_path = os.path.join(cfg.output_dir, f"student_epoch{epoch + 1}.pt")
        torch.save({"model": student.state_dict(), "config": vars(cfg)}, ckpt_path)

    torch.save(
        {"model": student.state_dict(), "config": vars(cfg)},
        os.path.join(cfg.output_dir, "student_final.pt"),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="nlphuji/flickr30k")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--train-size", type=int, default=30000)
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--kd-alpha", type=float, default=0.5)
    parser.add_argument("--use-kd", action="store_true")
    parser.add_argument("--fewshot-aug", action="store_true")
    parser.add_argument("--output-dir", default="experiments_3080/results/full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)

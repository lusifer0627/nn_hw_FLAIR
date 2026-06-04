import argparse
import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import open_clip


@dataclass
class TrainConfig:
    dataset_name: str = "nlphuji/flickr30k"
    train_size: int = 30000
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-5
    temperature: float = 0.07
    kd_alpha: float = 0.5
    use_kd: bool = False
    fewshot_aug: bool = False
    output_dir: str = "experiments_3080/results/full"
    seed: int = 42


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
            caption = random.choice/templates if False else random.choice(templates).format(caption)

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

    return (loss_i2t + loss_t2i) / 2, logits


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


def train(cfg: TrainConfig):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Student：用較小或同型 CLIP-style 模型訓練
    student, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained="openai",
        device=device,
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    # Teacher：這裡先用 OpenCLIP ViT-B/16 當可直接跑的 teacher。
    # 若你已能在 FLAIR repo 載入 flair-cc3m-recap.pt，可把 teacher 換成 FLAIR model。
    teacher, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained="laion2b_s34b_b88k",
        device=device,
    )
    freeze_teacher(teacher)

    ds = load_dataset(cfg.dataset_name, split="test")

    total_size = min(cfg.train_size, len(ds))
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:total_size]

    train_ds = Flickr30KDataset(
        Subset(ds, indices),
        preprocess=preprocess,
        fewshot_aug=cfg.fewshot_aug,
    )

    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    student.train()

    for epoch in range(cfg.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        total_loss = 0.0

        for images, captions in pbar:
            images = images.to(device)
            texts = tokenizer(list(captions)).to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=True):
                s_img = student.encode_image(images)
                s_txt = student.encode_text(texts)

                ce_loss, _ = contrastive_loss(
                    s_img,
                    s_txt,
                    temperature=cfg.temperature,
                )

                if cfg.use_kd:
                    with torch.no_grad():
                        t_img = teacher.encode_image(images)
                        t_txt = teacher.encode_text(texts)

                    distill = kd_loss(
                        s_img,
                        s_txt,
                        t_img,
                        t_txt,
                        temperature=cfg.temperature,
                    )

                    loss = (1 - cfg.kd_alpha) * ce_loss + cfg.kd_alpha * distill
                else:
                    loss = ce_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

        ckpt_path = os.path.join(cfg.output_dir, f"student_epoch{epoch+1}.pt")
        torch.save(
            {
                "model": student.state_dict(),
                "config": vars(cfg),
            },
            ckpt_path,
        )

    torch.save(
        {
            "model": student.state_dict(),
            "config": vars(cfg),
        },
        os.path.join(cfg.output_dir, "student_final.pt"),
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-name", default="nlphuji/flickr30k")
    parser.add_argument("--train-size", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--kd-alpha", type=float, default=0.5)
    parser.add_argument("--use-kd", action="store_true")
    parser.add_argument("--fewshot-aug", action="store_true")
    parser.add_argument("--output-dir", default="experiments_3080/results/full")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
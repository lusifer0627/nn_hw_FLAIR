import argparse
import json
import random

import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import open_clip


def build_eval_indices(dataset_len: int, seed: int, max_samples: int):
    """Use the same fixed holdout rule as train.py.

    train.py holds out the first eval_size shuffled indices. Here max_samples
    should match train.py's --eval-size, so evaluation is non-overlapping.
    """
    if max_samples <= 0:
        raise ValueError("max_samples must be > 0")
    if max_samples >= dataset_len:
        raise ValueError(f"max_samples={max_samples} must be smaller than dataset size={dataset_len}")

    rng = random.Random(seed)
    all_indices = list(range(dataset_len))
    rng.shuffle(all_indices)
    return all_indices[:max_samples]


class FlickrEvalDataset(Dataset):
    def __init__(self, hf_dataset, preprocess):
        self.ds = hf_dataset
        self.preprocess = preprocess

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        captions = item.get("caption", None)
        if captions is None:
            captions = item.get("captions", None)

        if isinstance(captions, list):
            caption = captions[0]
        else:
            caption = str(captions)

        return self.preprocess(image), caption


def recall_at_k(similarity, k):
    n = similarity.size(0)
    labels = torch.arange(n, device=similarity.device)

    i2t_topk = similarity.topk(k, dim=1).indices
    i2t = (i2t_topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    t2i_topk = similarity.T.topk(k, dim=1).indices
    t2i = (t2i_topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    return i2t, t2i


@torch.no_grad()
def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained="openai",
        device=device,
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    ckpt_config = None
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        ckpt_config = ckpt.get("config", None)

    model.eval()

    # If the checkpoint stores the same dataset settings, reuse them by default.
    dataset_name = args.dataset_name
    dataset_split = args.dataset_split
    seed = args.seed
    if ckpt_config is not None:
        dataset_name = ckpt_config.get("dataset_name", dataset_name)
        dataset_split = ckpt_config.get("dataset_split", dataset_split)
        seed = ckpt_config.get("seed", seed)

    ds = load_dataset(dataset_name, split=dataset_split)
    eval_indices = build_eval_indices(
        dataset_len=len(ds),
        seed=seed,
        max_samples=args.max_samples,
    )

    print(f"Dataset: {dataset_name} / split={dataset_split}")
    print(f"Dataset size: {len(ds)}")
    print(f"Eval size: {len(eval_indices)}")
    print(f"Seed: {seed}")
    print(f"Eval indices preview: {eval_indices[:10]}")

    eval_ds = FlickrEvalDataset(Subset(ds, eval_indices), preprocess)

    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )

    image_features = []
    text_features = []

    for images, captions in tqdm(loader, desc="Extracting features"):
        images = images.to(device, non_blocking=True)
        texts = tokenizer(list(captions)).to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            img = model.encode_image(images)
            txt = model.encode_text(texts)

        img = F.normalize(img.float(), dim=-1)
        txt = F.normalize(txt.float(), dim=-1)

        image_features.append(img)
        text_features.append(txt)

    image_features = torch.cat(image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)
    similarity = image_features @ text_features.T

    results = {}
    for k in [1, 5, 10]:
        i2t, t2i = recall_at_k(similarity, k)
        results[f"I2T_R@{k}"] = i2t
        results[f"T2I_R@{k}"] = t2i
        print(f"R@{k} | I2T: {i2t:.4f} | T2I: {t2i:.4f}")

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to: {args.save_json}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="nlphuji/flickr30k")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-json", default="")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())

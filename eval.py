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
    """
    Use the same fixed holdout rule as train.py.

    train.py holds out the first eval_size shuffled indices.
    Here max_samples should match train.py's --eval-size,
    so evaluation is non-overlapping.
    """
    if max_samples <= 0:
        raise ValueError("max_samples must be > 0")
    if max_samples >= dataset_len:
        raise ValueError(
            f"max_samples={max_samples} must be smaller than dataset size={dataset_len}"
        )

    rng = random.Random(seed)
    all_indices = list(range(dataset_len))
    rng.shuffle(all_indices)
    return all_indices[:max_samples]


class FlickrEvalDataset(Dataset):
    """
    Evaluation dataset.

    Important:
    - Training can randomly choose one caption per image.
    - Evaluation should keep all captions per image.
    - Flickr30k / COCO usually have 5 captions per image.
    """

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

        if captions is None:
            raise KeyError("Cannot find caption or captions field in dataset item.")

        if isinstance(captions, list):
            captions = [str(c) for c in captions]
        else:
            captions = [str(captions)]

        return self.preprocess(image), captions


def collate_eval(batch):
    """
    Custom collate function because each image has a list of captions.
    """
    images = torch.stack([x[0] for x in batch], dim=0)
    captions_per_image = [x[1] for x in batch]
    return images, captions_per_image


def recall_i2t(similarity: torch.Tensor, image_to_text: list[list[int]], k: int) -> float:
    """
    Image-to-text retrieval.

    similarity:
        Tensor with shape [num_images, num_texts]

    image_to_text:
        image_to_text[i] is a list of correct caption indices for image i.

    Rule:
        For each image, if any correct caption appears in top-k, it is a hit.
    """
    topk = similarity.topk(k, dim=1).indices.cpu().tolist()

    hits = 0
    for img_idx, retrieved_texts in enumerate(topk):
        positives = set(image_to_text[img_idx])
        if any(text_idx in positives for text_idx in retrieved_texts):
            hits += 1

    return hits / len(image_to_text)


def recall_t2i(similarity: torch.Tensor, text_to_image: list[int], k: int) -> float:
    """
    Text-to-image retrieval.

    similarity:
        Tensor with shape [num_images, num_texts]

    text_to_image:
        text_to_image[t] is the correct image index for caption t.

    Rule:
        For each caption, if its correct image appears in top-k, it is a hit.
    """
    topk = similarity.T.topk(k, dim=1).indices
    targets = torch.tensor(text_to_image, device=topk.device)

    hits = (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()
    return hits


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

        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
            ckpt_config = ckpt.get("config", None)
        else:
            model.load_state_dict(ckpt, strict=False)

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
    print(f"Eval image size: {len(eval_indices)}")
    print(f"Seed: {seed}")
    print(f"Eval indices preview: {eval_indices[:10]}")

    eval_ds = FlickrEvalDataset(Subset(ds, eval_indices), preprocess)

    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_amp,
        collate_fn=collate_eval,
    )

    image_features = []
    text_features = []

    image_to_text = []
    text_to_image = []

    num_images_seen = 0
    num_texts_seen = 0

    for images, captions_per_image in tqdm(loader, desc="Extracting features"):
        images = images.to(device, non_blocking=True)

        flat_captions = []

        for local_img_idx, caps in enumerate(captions_per_image):
            global_img_idx = num_images_seen + local_img_idx

            current_text_indices = list(
                range(num_texts_seen, num_texts_seen + len(caps))
            )
            image_to_text.append(current_text_indices)

            for cap in caps:
                flat_captions.append(cap)
                text_to_image.append(global_img_idx)

            num_texts_seen += len(caps)

        texts = tokenizer(flat_captions).to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            img = model.encode_image(images)
            txt = model.encode_text(texts)

        img = F.normalize(img.float(), dim=-1)
        txt = F.normalize(txt.float(), dim=-1)

        image_features.append(img)
        text_features.append(txt)

        num_images_seen += images.size(0)

    image_features = torch.cat(image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)

    similarity = image_features @ text_features.T

    print("\nFeature / similarity info")
    print(f"Image features: {tuple(image_features.shape)}")
    print(f"Text features: {tuple(text_features.shape)}")
    print(f"Similarity matrix: {tuple(similarity.shape)}")
    print(f"Number of images: {len(image_to_text)}")
    print(f"Number of texts: {len(text_to_image)}")
    print(f"Avg captions per image: {len(text_to_image) / len(image_to_text):.2f}")

    results = {
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "num_images": len(image_to_text),
        "num_texts": len(text_to_image),
        "avg_captions_per_image": len(text_to_image) / len(image_to_text),
    }

    print("\nRetrieval results")
    for k in [1, 5, 10]:
        i2t = recall_i2t(similarity, image_to_text, k)
        t2i = recall_t2i(similarity, text_to_image, k)

        results[f"I2T_R@{k}"] = i2t
        results[f"T2I_R@{k}"] = t2i

        print(f"R@{k} | I2T: {i2t:.4f} | T2I: {t2i:.4f}")

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved results to: {args.save_json}")


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
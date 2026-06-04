import argparse
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import open_clip


class FlickrEvalDataset(Dataset):
    def __init__(self, hf_dataset, preprocess, max_samples=1000):
        self.ds = hf_dataset
        self.preprocess = preprocess
        self.max_samples = min(max_samples, len(hf_dataset))

    def __len__(self):
        return self.max_samples

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
    # similarity: image x text
    n = similarity.size(0)
    labels = torch.arange(n, device=similarity.device)

    # image-to-text
    i2t_topk = similarity.topk(k, dim=1).indices
    i2t = (i2t_topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    # text-to-image
    t2i_topk = similarity.T.topk(k, dim=1).indices
    t2i = (t2i_topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    return i2t, t2i


@torch.no_grad()
def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained="openai",
        device=device,
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)

    model.eval()

    ds = load_dataset(args.dataset_name, split="test")
    eval_ds = FlickrEvalDataset(ds, preprocess, max_samples=args.max_samples)

    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    image_features = []
    text_features = []

    for images, captions in tqdm(loader, desc="Extracting features"):
        images = images.to(device)
        texts = tokenizer(list(captions)).to(device)

        img = model.encode_image(images)
        txt = model.encode_text(texts)

        img = F.normalize(img, dim=-1)
        txt = F.normalize(txt, dim=-1)

        image_features.append(img)
        text_features.append(txt)

    image_features = torch.cat(image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)

    similarity = image_features @ text_features.T

    for k in [1, 5, 10]:
        i2t, t2i = recall_at_k(similarity, k)
        print(f"R@{k} | I2T: {i2t:.4f} | T2I: {t2i:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="nlphuji/flickr30k")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
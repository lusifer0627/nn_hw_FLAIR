# nn_hw_FLAIR

## 1. Clone 專案後解壓縮並建立環境
```bash 
git clone https://github.com/lusifer0627/nn_hw_FLAIR.git
cd flair

python3.12 -m venv flair_env
source flair_env/bin/activate
```

## 2. 安裝套件
```bash 
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 3. 準備 Flickr30k 資料集
下載[Flickr30k 資料集](https://huggingface.co/datasets/nlphuji/flickr30k)到/mnt/d/M11417018/hw_115/nn_hw/datasets/flickr30k-images

檔案結構:
```
~/datasets/flickr30k-images/
├── 1000092795.jpg
├── ...
├── flickr30k_val.json
└── flickr30k_test.json
```

## 4. 修改inference.sh
```bash 
torchrun --nproc_per_node 1 -m main \
    --model ViT-B-16-FLAIR \
    --huggingface-repo-name xiaorui638/flair \
    --huggingface-model-name flair-merged30m.pt \
    --inference-with-flair \
    --flickr-data-root-dir /Flickr30k資料集檔案路徑 \
    --retrieval-flickr \
    --batch-size 128 \
    --precision amp \
    --workers 4
```

## 5. 執行推論
```bash 
bash inference.sh
```

R@1 / R@5結果如下:
```json 
{"retrieval_flickr_text_to_image_R@1": 0.8185404339250493, "retrieval_flickr_text_to_image_R@5": 0.9506903353057199, "retrieval_flickr_text_to_image_R@10": 0.9741617357001973, "retrieval_flickr_text_to_image_mean_rank": 2.4376726150512695, "retrieval_flickr_text_to_image_median_rank": 1.0, "retrieval_flickr_image_to_text_R@1": 0.9516765285996055, "retrieval_flickr_image_to_text_R@5": 0.995069033530572, "retrieval_flickr_image_to_text_R@10": 0.9970414201183432, "retrieval_flickr_image_to_text_mean_rank": 1.1844181418418884, "retrieval_flickr_image_to_text_median_rank": 1.0, "epoch": 0, "retrieval_flickr_num_text_samples": 5070, "retrieval_flickr_num_image_samples": 1014}
```

## 6. 下載CIFAR-10資料集
下載[CIFAR-10 資料集](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

## 7. 建立eval_cifar10_top1.py
```python 
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import flair


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# 多個 prompt template 平均，會比只用一個 prompt 穩定
TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of the {}.",
    "a close-up photo of a {}.",
]


def build_prompts():
    prompts = []
    for class_name in CIFAR10_CLASSES:
        for template in TEMPLATES:
            prompts.append(template.format(class_name))
    return prompts


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/mnt/d/M11417018/hw_115/nn_hw/datasets")
    parser.add_argument("--model-name", type=str, default="flair-merged30m.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Model:", args.model_name)

    # 下載或載入 FLAIR 權重
    pretrained = flair.download_weights_from_hf(
        model_repo="xiaorui638/flair",
        filename=args.model_name,
    )

    model, _, preprocess = flair.create_model_and_transforms(
        "ViT-B-16-FLAIR",
        pretrained=pretrained,
    )
    model = model.to(device)
    model.eval()

    tokenizer = flair.get_tokenizer("ViT-B-16-FLAIR")

    dataset = CIFAR10(
        root=args.data_root,
        train=False,
        download=True,
        transform=preprocess,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    prompts = build_prompts()
    text = tokenizer(prompts).to(device)

    num_classes = len(CIFAR10_CLASSES)
    num_templates = len(TEMPLATES)

    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model.get_logits(image=images, text=text)

        # FLAIR / OpenCLIP 有時會回傳 tuple: (image_logits, text_logits)
        if isinstance(logits, tuple):
            logits = logits[0]

        # logits: [batch, num_classes * num_templates]
        logits = logits.view(images.size(0), num_classes, num_templates)

        # 同一類的多個 prompt 分數取平均
        logits = logits.mean(dim=2)

        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if total % 1000 == 0:
            print(f"Processed {total}/{len(dataset)} images")

    acc = correct / total * 100
    print("=" * 50)
    print(f"CIFAR-10 Top-1 Accuracy: {acc:.2f}%")
    print(f"Correct: {correct} / {total}")
    print("=" * 50)


if __name__ == "__main__":
    main()
```

## 8. 終端直接執行腳本推論
```bash 
python eval_cifar10_top1.py \
    --data-root /CIFAR-10資料集路徑 \
    --model-name flair-merged30m.pt \
    --batch-size 128 \
    --workers 4
```

CIFAR-10 Top-1 Accuracy結果如下:
```
Processed 10000/10000 images
==================================================
CIFAR-10 Top-1 Accuracy: 92.82%
Correct: 9282 / 10000
==================================================
```

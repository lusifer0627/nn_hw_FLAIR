# nn_hw_final_project
* [Flair GitHub](https://github.com/ExplainableML/flair)

* 由於模型所使用之資料集CC3M過於龐大，且使用8張A100進行訓練，單靠單張RTX 3080無法有效復現出論文中的效能，因此希望使用較小規模的Flickr30K進行訓練集驗證，並且驗證資料量少的情況下，加入少樣本學習方法對於模型效能的影響。

# 使用方法比較
|方法|訓練資料量|說明|
|:-:|:-:|:-:|
|flair pre-train model|30M|預訓練模型FLAIR 30M|
|Full Data|Flickr30K|多資料 baseline|
|Low Data|Flickr1K|少資料 baseline|
|Low Data + Knowledge Distillation|Flickr1K|少資料知識蒸餾|
|Low Data + Knowledge Distillation + Few shot|Flickr1K|少資料知識蒸餾 + Few shot|

## 1. Clone 專案並建立 WSL 環境
* 在WSL上運行

```bash 
git clone https://github.com/lusifer0627/nn_hw_FLAIR.git
cd flair

python3.12 -m venv my_env
source my_env/bin/activate
```

## 2. 安裝套件
```bash 
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 3. 進行訓練與驗證
### 1. Full Data 30K baseline
* 訓練

```python 
python train.py --train-size 30000 --eval-size 1000 --batch-size 16 --epochs 2 --lr 1e-6 --output-dir experiments_3080/results/full_30k
```

* 驗證

```python
python eval.py --ckpt experiments_3080/results/full_30k/student_final.pt --max-samples 1000 --batch-size 64 --seed 42
```

### 2. Low Data 1K baseline
* 訓練

```python
python train.py --train-size 1000 --eval-size 1000 --batch-size 16 --epochs 10 --lr 1e-5 --output-dir experiments_3080/results/low_1k
```

* 驗證

```python
python eval.py --ckpt experiments_3080/results/low_1k/student_final.pt --max-samples 1000 --batch-size 64 --seed 42
```

### 3. Low Data 1K + Knowledge Distillation
* 訓練

```python
python train.py --train-size 1000 --eval-size 1000 --batch-size 16 --epochs 5 --lr 1e-5 --use-kd --kd-alpha 0.5 --output-dir experiments_3080/results/low_1k_kd
```

* 驗證

```python
python eval.py --ckpt experiments_3080/results/low_1k_kd/student_final.pt --max-samples 1000 --batch-size 64 --seed 42
```

### 4. Low Data 1K + Knowledge Distillation + Few-shot
* 訓練

```python
python train.py --train-size 1000 --eval-size 1000 --batch-size 16 --epochs 10 --lr 5e-6 --use-kd --kd-alpha 0.7 --fewshot-aug --output-dir experiments_3080/results/low_1k_kd_fewshot
```

* 驗證

```python
python eval.py --ckpt experiments_3080/results/low_1k_kd_fewshot/student_final.pt --max-samples 1000 --batch-size 64 --seed 42
```

## 4. 執行結果
|方法|I2T R@1|I2T R@5|T2I R@1|T2I R@5|
|:-:|:-:|:-:|:-:|:-:|
|FLAIR 30M|94.7|99.3|81.1|94.9|
|Full Data|80.2|96.5|84.3|96.2|
|Low Data|70.8|90.7|72.9|92.7|
|Low Data + Knowledge Distillation|79.8|95.8|80.8|96.0|
|Low Data + Knowledge Distillation + Few-shot|80.7|96.2|83.5|96.2|

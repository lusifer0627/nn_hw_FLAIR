# nn_hw_FLAIR
* [Flair GitHub](https://github.com/ExplainableML/flair)

* 由於模型所使用之資料集CC3M過於龐大，且使用8張A100進行訓練，單靠單張RTX 3080無法有效復現出論文中的效能，因此希望使用較小規模的Flickr30K進行訓練集驗證，並且驗證資料量少的情況下，加入少樣本學習方法對於模型效能的影響。

## 1. Clone 專案並建立環境
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
```

## 3. 執行train.py進行訓練

## 4. 執行eval.py進行驗證

# nn_hw_FLAIR

1. Clone 專案並建立環境
```
git clone https://github.com/lusifer0627/nn_hw_FLAIR.git
cd flair

python3.12 -m venv flair_env
source flair_env/bin/activate
```

2. 安裝套件
```
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

3. 準備 Flickr30k 資料集

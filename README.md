## 環境構築

各自Anacondaで仮想環境を構築したのち以下のコマンドを入力してライブラリのインストールをしてください

```
pip install jupyterlab
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

環境が構築出来たら以下のコマンドでjupyter labを起動してください
```
jupyter lab
```

### gitの基礎コマンド
更新したファイルをすべてステージする
```
git add .
```

更新したファイルをコミットする
```
git commit -m 'コミットメッセージ'
```

リモートにpushする
```
git push origin ブランチ名
```

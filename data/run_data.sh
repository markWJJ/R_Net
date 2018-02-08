#/bin/bash

if [ ! -f "train-v1.1.json" ];then
    echo "下载数据 train-v1.1"
    wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
else
    echo "train-v1.1 已经存在"
fi

if [ ! -f "dev-v1.1.json"];then
    echo "下载数据 dev-v1.1"
    wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
else
    echo "dev-v1.1 已经存在"
fi

echo "数据提取....."
python data_convert.py
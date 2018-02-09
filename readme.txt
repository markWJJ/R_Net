R_net问答模型 论文地址：https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf

数据集：
数据集为 机器阅读理解问答数据集：SQuAD
下载地址-train：https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
下载地址-dev：https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json


训练数据文件：train_out_*.txt 分别有20样本/200样本/2K样本/5K样本/2w样本
验证数据文件：dev_out.txt 有2W数据
测试数据文件：test_out.txt

首先 下载模型文件：https://pan.baidu.com/s/1c4ey0pE  将模型文件放于 save_model/下
然后运行 sh data_download.sh
随后进行训练活在预测

训练：
>> python R_Net.py --mod=train --sample_num=200 #默认为200的样本集
模型保存地址为 ./save_model 目前训练好的模型为200样本

预测：
>> python R_Net.py --mod=infer --test_dir=./data/test.txt

需求：
Tensorflow-gpu>=1.0.0


模块分析：
./data/data_convert.py #数据提取模型：将原始json文件转换为txt的模型输入文件
./save_model/    #模型保存文件
./log/   #模型loss的可视化log的保存路径
data_prepocess.py #模型的数据处理模块,读取输入的txt文件，进行分词,词典构造，shuffle，输入构建等功能
R_Net.py  #模型构建模块
vocab.p  #训练数据/测试数据/验证数据 构建的词典



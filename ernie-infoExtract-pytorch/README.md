# 百度2021多形态信息抽取（实体属性/关系三元组抽取）官方baseline模型-Pytorch

----------------
联合抽取模型

## 1. 目录结构
------------------------
    ./data/ # 存放数据，包括train_data.json、dev_data.json以及test_data.json
    ./run_duie.py # 模型及训练、测试
    ./data_loader.py # 数据加载
    ./utils.py # 解码等相关
    ./extract_chinese_and_punct.py # 判断中文字符
    ./re_official_evaluation.py # 评估指标

## 2. 第三方库
-------------------------
transformers4.4.2
pytorch1.4.0或以上

## 3. 运行
-------------------------
    python run_duie.py

## 4. 数据
--------------------------------
数据获取：https://aistudio.baidu.com/aistudio/competition/detail/65?isFromLuge=true

---------------------------------
Reference: https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/information_extraction/DuIE
# 百度2021多形态信息抽取（实体关系/属性联合抽取）-SPN关系抽取模型

## 1. 论文来源
--------------------------------

论文：[Joint Entity and Relation Extraction with Set Prediction Networks](https://arxiv.org/abs/2011.01675v1)
作者：Dianbo Sui, Yubo Chen, Kang Liu, Jun Zhao, Xiangrong Zeng, Shengping Liu
优势：解决实体重叠问题

## 2. 目录结构
--------------------------
    ./data/
        data-chinese/
            train_data.json # 训练集
            valid_data.json # 验证集
            test_data.json # 测试集
        generated_data/
            model_param/ # 存放模型
            *.pickle # 预处理中间数据
        submit/ # 存放预测结果
    ./model/
        __init__.py
        matcher.py # 二部图匹配
        seq_encoder.py # bert编码器
        set_criterion.py # 损失模型
        set_decoder.py # 解码器
        setpred4RE.py # SPN模型
    ./trainer/
        __init__.py
        trainer.py # 训练和预测
    ./utils/
        __init__.py
        alphabet.py # 关系2id
        average_meter.py # 记录
        data.py # 数据
        dataloader.py # 数据批加载
        decoding.py # 文本解码
        extract_chinese_and_punct.py # 中文字符检测
        functions.py # 数据预处理
        metric.py # 评价指标
    main.py

## 3. 第三方库
---------------------------
transformers4.4.2
pytorch1.4.0或以上

## 4. 运行
---------------------------
    python main.py

## 数据
---------------------------
数据获取：https://aistudio.baidu.com/aistudio/competition/detail/65?isFromLuge=true

---------------------------
Reference: https://github.com/DianboWork/SPN4RE
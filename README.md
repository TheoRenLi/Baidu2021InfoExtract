# 百度2021多形态信息抽取

## 1. 任务
--------------------
    (1) 实体关系/属性抽取
    (2) 句子级事件抽取
    (3) 篇章级事件抽取

## 2. 目录
--------------------
    (1) 官方baseline模型-pytorch
        - ernie-infoExtract-pytorch # 信息抽取（联合抽取模型）
        - ernie-eventExtract-pytroch # 事件抽取（pipeline模型）

    (2) SPN4RE模型
        - SPN4RE-infoExtract # 参赛使用的模型（联合抽取模型）

## 3. 数据来源
--------------------------------
数据获取：https://aistudio.baidu.com/aistudio/competition/detail/65?isFromLuge=true

--------------------------------
Reference:
    - https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/information_extraction
    - https://github.com/DianboWork/SPN4RE
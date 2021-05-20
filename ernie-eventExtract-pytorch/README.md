# 百度2021多形态信息抽取（事件抽取）官方baseline模型-Pytorch

---------------------------
pipeline方法

## 1. 目录结构
----------------------------
    ./conf/
        DuEE1.0/event_schema.json # 句子级事件抽取schema
        DuEE-Fin/event_schema.json # 篇章级事件抽取schema
    ./data/
        DuEE1.0/ # 存放数据，包括train_data.json、dev_data.json和test_data.json
        DuEE-Fin/ # 存放数据，包括train_data.json、dev_data.json和test_data.json
    ./chunk.py # 评估指标
    ./classifier.py # 篇章级事件抽取-文本分类
    ./duee_1_data_prepare,py # 句子级事件抽取-数据预处理
    ./duee_1_data_postprocess.py # 句子级事件抽取-数据后处理
    ./duee_fin_data_prepare.py # 篇章级事件抽取-数据预处理
    ./duee_fin_data_postprocess.py # 篇章级事件抽取-数据后处理
    ./metrics.py # 评估指标
    ./sequence_labeling.py # 事件抽取-序列标注
    ./stack_pad_tuple.py # 批数据拼接处理等
    ./utils.py # 工具函数

## 2. 第三方库
------------------------------
transformers4.4.2
pytorch1.4.0或以上

## 3. 运行
------------------------------
    # 句子级事件抽取
    bash run_duee_1.sh data_prepare # 数据预处理
    bash run_duee_1.sh trigger_train # 触发词标注训练
    bash run_duee_1.sh trigger_predict # 触发词标注预测
    bash run_duee_1.sh role_train # 论元标注训练
    bash run_duee_1.sh role_predict # 论元标注预测
    bash run_duee_1.sh pred_2_submit # 事件类型和论元匹配
    
    # 篇章级事件抽取
    bash run_duee_fin.sh data_prepare # 数据预处理
    bash run_duee_fin.sh trigger_train # 触发词标注训练
    bash run_duee_fin.sh trigger_predict # 触发词标注预测
    bash run_duee_fin.sh role_train # 论元标注训练
    bash run_duee_fin.sh role_predict # 论元标注预测
    bash run_duee_fin.sh enum_train # 事件分类训练
    bash run_duee_fin.sh enum_predict # 事件分类预测
    bash run_duee_fin.sh pred_2_submit # 事件类型和论元匹配

## 4. 数据
--------------------------------
数据获取：https://aistudio.baidu.com/aistudio/competition/detail/65?isFromLuge=true


--------------------------------
Reference: https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/information_extraction/DuEE
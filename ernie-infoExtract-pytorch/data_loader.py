import collections
import json
import os
from typing import Optional, List, Union, Dict # 该模块用于检查函数参数和返回值类型是否正确，Optional[X]声明的变量可以是None，特别是对于（a: int = None）这种类型。其相当于Union[X, None]
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer

from extract_chinese_and_punct import ChineseAndPunctuationExtractor

InputFeature = collections.namedtuple("InputFeature", [
    "input_ids", "pos_ids", "seq_len", "tok_to_orig_start_index", "tok_to_orig_end_index",
    "labels"
])


def parse_label(spo_list, label_map, tokens, tokenizer):
    # 2 tags for each predicate + I tag + O tag
    num_labels = 2 * (len(label_map.keys()) - 2) + 2
    seq_len = len(tokens)
    # initialize tag
    labels = [[0] * num_labels for i in range(seq_len)] # 这里单个label的维度为112d，0表示标注O，1表示标注I，2-56是对subject的标注，57-112是对object的标注
    #  find all entities and tag them with corresponding "B"/"I" labels
    for spo in spo_list: # 该循环把实体的label标注好(subject的实体标注idx处于2到56， object的实体标注处于57到112之间，B标注)
        for spo_object in spo['object'].keys():
            # assign relation label
            if spo['predicate'] in label_map.keys():
                # simple relation
                label_subject = label_map[spo['predicate']]
                label_object = label_subject + 55 # why 55 ?
                subject_tokens = tokenizer.tokenize(spo['subject'])
                object_tokens = tokenizer.tokenize(spo['object']['@value'])
            else:
                # complex relation
                label_subject = label_map[spo['predicate'] + '_' + spo_object]
                label_object = label_subject + 55
                subject_tokens = tokenizer.tokenize(spo['subject'])
                object_tokens = tokenizer.tokenize(spo['object'][spo_object])

            subject_tokens_len = len(subject_tokens)
            object_tokens_len = len(object_tokens)

            # assign token label
            # there are situations where s entity and o entity might overlap, e.g. xyz established xyz corporation
            # to prevent single token from being labeled into two different entity
            # we tag the longer entity first, then match the shorter entity within the rest text
            # BIO标注的策略：例句：李明是李晓之的父亲。
            # 李明标注为BI，李晓之标注为BII，其他标注为O
            forbidden_index = None
            if subject_tokens_len > object_tokens_len:
                for index in range(seq_len - subject_tokens_len + 1):
                    if tokens[index:index +
                              subject_tokens_len] == subject_tokens: # 查找到了subject_token
                        labels[index][label_subject] = 1 # 给subject实体的开头标注B
                        for i in range(subject_tokens_len - 1): # 给subject实体的其他部位标注I
                            labels[index + i + 1][1] = 1
                        forbidden_index = index
                        break

                for index in range(seq_len - object_tokens_len + 1):
                    if tokens[index:index + object_tokens_len] == object_tokens: # 查找到了object_token
                        if forbidden_index is None:
                            labels[index][label_object] = 1
                            for i in range(object_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            break
                        # check if labeled already
                        elif index < forbidden_index or index >= forbidden_index + len(
                                subject_tokens):
                            labels[index][label_object] = 1
                            for i in range(object_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            break

            else:
                for index in range(seq_len - object_tokens_len + 1):
                    if tokens[index:index + object_tokens_len] == object_tokens:
                        labels[index][label_object] = 1
                        for i in range(object_tokens_len - 1):
                            labels[index + i + 1][1] = 1
                        forbidden_index = index
                        break

                for index in range(seq_len - subject_tokens_len + 1):
                    if tokens[index:index +
                              subject_tokens_len] == subject_tokens:
                        if forbidden_index is None:
                            labels[index][label_subject] = 1
                            for i in range(subject_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            break
                        elif index < forbidden_index or index >= forbidden_index + len(
                                object_tokens):
                            labels[index][label_subject] = 1
                            for i in range(subject_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            break

    # if token wasn't assigned as any "B"/"I" tag, give it an "O" tag for outside
    for i in range(seq_len): # 该循环把非实体的字标注好（[1, 0, 0, 0, ..., 0] 1d x 112d）
        if labels[i] == [0] * num_labels:
            labels[i][0] = 1

    return labels


def convert_example_to_feature(
        example,
        tokenizer: AutoTokenizer,
        chineseandpunctuationextractor: ChineseAndPunctuationExtractor,
        label_map,
        max_length: Optional[int]=512,
        pad_to_max_length: Optional[bool]=None):
    spo_list = example['spo_list'] if "spo_list" in example.keys() else None
    text_raw = example['text']

    sub_text = [] # 放置中文字符
    buff = "" # 存放非中文字符
    for char in text_raw:
        if chineseandpunctuationextractor.is_chinese_or_punct(char):
            if buff != "":
                sub_text.append(buff)
                buff = ""
            sub_text.append(char)
        else:
            buff += char
    if buff != "":
        sub_text.append(buff)
    

    tok_to_orig_start_index = []
    tok_to_orig_end_index = []
    orig_to_tok_index = []
    tokens = []
    text_tmp = ''
    for (i, token) in enumerate(sub_text):
        orig_to_tok_index.append(len(tokens))
        sub_tokens = tokenizer.tokenize(token)
        text_tmp += token
        for sub_token in sub_tokens:
            tok_to_orig_start_index.append(len(text_tmp) - len(token))
            tok_to_orig_end_index.append(len(text_tmp) - 1)
            tokens.append(sub_token)
            if len(tokens) >= max_length - 2:
                break
        else:
            continue
        break
    # print("tok_to_orig_start_index: ", tok_to_orig_start_index)
    # print("tok_to_orig_end_index: ", tok_to_orig_end_index)
    # print("orig_to_tok_index: ", orig_to_tok_index)
    # print("tokens: ", tokens)
    seq_len = len(tokens)
    # 2 tags for each predicate + I tag + O tag
    num_labels = 2 * (len(label_map.keys()) - 2) + 2
    # initialize tag
    labels = [[0] * num_labels for i in range(seq_len)] # 每个字都要生成标签表示，用于预测

    if spo_list is not None:
        labels = parse_label(spo_list, label_map, tokens, tokenizer)

    # add [CLS] and [SEP] token, they are tagged into "O" for outside
    if seq_len > max_length - 2:
        tokens = tokens[0:(max_length - 2)]
        labels = labels[0:(max_length - 2)]
        tok_to_orig_start_index = tok_to_orig_start_index[0:(max_length - 2)]
        tok_to_orig_end_index = tok_to_orig_end_index[0:(max_length - 2)]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    # "O" tag for [PAD], [CLS], [SEP] token
    outside_label = [[1] + [0] * (num_labels - 1)]
    
    labels = outside_label + labels + outside_label
    tok_to_orig_start_index = [-1] + tok_to_orig_start_index + [-1]
    tok_to_orig_end_index = [-1] + tok_to_orig_end_index + [-1]
    if seq_len < max_length:
        tokens = tokens + ["[PAD]"] * (max_length - seq_len - 2)
        labels = labels + outside_label * (max_length - len(labels))
        tok_to_orig_start_index = tok_to_orig_start_index + [-1] * (
            max_length - len(tok_to_orig_start_index))
        tok_to_orig_end_index = tok_to_orig_end_index + [-1] * (
            max_length - len(tok_to_orig_end_index))

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    return InputFeature(
        input_ids=np.array(token_ids),
        seq_len=np.array(seq_len),
        tok_to_orig_start_index=np.array(tok_to_orig_start_index),
        tok_to_orig_end_index=np.array(tok_to_orig_end_index),
        labels=np.array(labels), )


class DuIEDataset(Dataset):
    """
    Dataset of DuIE.
    """

    def __init__(
            self,
            input_ids: List[Union[List[int], np.ndarray]],
            seq_lens: List[Union[List[int], np.ndarray]],
            tok_to_orig_start_index: List[Union[List[int], np.ndarray]],
            tok_to_orig_end_index: List[Union[List[int], np.ndarray]],
            labels: List[Union[List[int], np.ndarray, List[str], List[Dict]]]):
        super(DuIEDataset, self).__init__()

        self.input_ids = input_ids
        self.seq_lens = seq_lens
        self.tok_to_orig_start_index = tok_to_orig_start_index
        self.tok_to_orig_end_index = tok_to_orig_end_index
        self.labels = labels

    def __len__(self):
        if isinstance(self.input_ids, np.ndarray):
            return self.input_ids.shape[0]
        else:
            return len(self.input_ids)

    def __getitem__(self, item):
        return {
            "input_ids": np.array(self.input_ids[item]),
            "seq_lens": np.array(self.seq_lens[item]),
            "tok_to_orig_start_index":
            np.array(self.tok_to_orig_start_index[item]),
            "tok_to_orig_end_index": np.array(self.tok_to_orig_end_index[item]),
            # If model inputs is generated in `collate_fn`, delete the data type casting.
            "labels": np.array(
                self.labels[item], dtype=np.float32),
        }

    @classmethod
    def from_file(cls,
                  file_path: Union[str, os.PathLike],
                  tokenizer: AutoTokenizer,
                  max_length: Optional[int]=512,
                  pad_to_max_length: Optional[bool]=None):
        assert os.path.exists(file_path) and os.path.isfile(
            file_path), f"{file_path} dose not exists or is not a file."
        label_map_path = os.path.join(
            os.path.dirname(file_path), "predicate2id.json")
        assert os.path.exists(label_map_path) and os.path.isfile(
            label_map_path
        ), f"{label_map_path} dose not exists or is not a file."
        with open(label_map_path, 'r', encoding='utf8') as fp:
            label_map = json.load(fp)
        chineseandpunctuationextractor = ChineseAndPunctuationExtractor()

        input_ids, seq_lens, tok_to_orig_start_index, tok_to_orig_end_index, labels = (
            [] for _ in range(5))
        dataset_scale = sum(1 for line in open(file_path, 'r', encoding='utf8'))
        print("Preprocessing data, loaded from %s" % file_path)
        with open(file_path, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in tqdm(lines):
                example = json.loads(line)
                input_feature = convert_example_to_feature(
                    example, tokenizer, chineseandpunctuationextractor,
                    label_map, max_length, pad_to_max_length)
                input_ids.append(input_feature.input_ids)
                seq_lens.append(input_feature.seq_len)
                tok_to_orig_start_index.append(
                    input_feature.tok_to_orig_start_index)
                tok_to_orig_end_index.append(
                    input_feature.tok_to_orig_end_index)
                labels.append(input_feature.labels)

        return cls(input_ids, seq_lens, tok_to_orig_start_index,
                   tok_to_orig_end_index, labels)


@dataclass
class DataCollator:
    """
    Collator for DuIE.
    """

    def __call__(self, examples: List[Dict[str, Union[list, np.ndarray]]]):
        batched_input_ids = torch.from_numpy(np.stack([x['input_ids'] for x in examples])).long()
        seq_lens = torch.from_numpy(np.stack([x['seq_lens'] for x in examples]))
        tok_to_orig_start_index = torch.from_numpy(np.stack(
            [x['tok_to_orig_start_index'] for x in examples]))
        tok_to_orig_end_index = torch.from_numpy(np.stack(
            [x['tok_to_orig_end_index'] for x in examples]))
        labels = torch.from_numpy(np.stack([x['labels'] for x in examples]))

        return (batched_input_ids, seq_lens, tok_to_orig_start_index,
                tok_to_orig_end_index, labels)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
    data = DuIEDataset.from_file("./data/train_data.json", tokenizer)
    

    collator = DataCollator()
    loader = DataLoader(dataset=data, batch_size=2, collate_fn=collator)

    for dd in loader:
        model_input = {
            "input_ids": dd[0],
            "seq_len": dd[1],
            "tok_to_orig_start_index": dd[2],
            "tok_to_orig_end_index": dd[3],
            "labels": dd[4]
        }
        print(model_input['input_ids'].dtype)
        print(model_input['seq_len'].dtype)
        print(model_input['tok_to_orig_start_index'].dtype)
        print(model_input['tok_to_orig_end_index'].dtype)
        print(model_input['labels'].dtype)
        break

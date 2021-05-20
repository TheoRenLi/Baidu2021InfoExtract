import torch, collections
from tqdm import tqdm
import json
import random
from utils.extract_chinese_and_punct import ChineseAndPunctuationExtractor


def count_most_possible(list1, list2):
    res = [True if list1[i] == list2[i] else False for i in range(min(len(list1), len(list2)))]
    res = collections.Counter(res)
    return res[True]


def list_index(list1: list, list2: list) -> list:
    try:
        start = [i for i, x in enumerate(list2) if x == list1[0]]
        end = [i for i, x in enumerate(list2) if x == list1[-1]]
    except Exception as e:
        # print("Error: ", e)
        return 0, 0
    if len(start) == 1 and len(end) == 1:
        if start[0] > end[0]:
            # print("Problem is here.")
            situa_1 = count_most_possible(list1, list2[start[0] : start[0]+len(list1)-1])
            situa_2 = count_most_possible(list1, list2[end[0]-len(list1)+1 : end[0]])
            if situa_1 > situa_2:
                return start[0], start[0] + len(list1) - 1
        else:
            return start[0], end[0]
    elif len(start) == 0 and len(end) != 0:
        # print("empty situation1.")
        pos, temp = -1, -1
        for j in end:
            count_ = count_most_possible(list1, list2[j-len(list1)+1:j])
            if count_ > temp:
                pos = j
                temp = count_
        return pos - len(list1) + 1, pos
    elif len(start) != 0 and len(end) == 0:
        # print("empty situation 2.")
        pos, temp = -1, -1
        for i in start:
            count_ = count_most_possible(list1, list2[i:i+len(list1)-1])
            if count_ > temp:
                pos = i
                temp = count_
        return pos, pos + len(list1) - 1
    elif len(start) == 0 and len(end) == 0:
        # print("empty situation 3.")
        return 0, 0
    else:
        for i in start:
            for j in end:
                if i <= j:
                    if list2[i:j+1] == list1:
                        index = (i, j)
                        break
            if 'index' in vars():
                break
        if not 'index' in vars():
            index = (start[0], start[0]+1)
            # print("tokens: ", list2)
            # print("entity: ", list1)
        return index[0], index[1]





# def list_index(list1: list, list2: list) -> list:
#     start = [i for i, x in enumerate(list2) if x == list1[0]]
#     end = [i for i, x in enumerate(list2) if x == list1[-1]]
#     if len(start) == 1 and len(end) == 1:
#         return start[0], end[0]
#     else:
#         for i in start:
#             for j in end:
#                 if i <= j:
#                     if list2[i:j+1] == list1:
#                         break
#         return i, j
#

def remove_accents(text: str) -> str:
    accents_translation_table = str.maketrans(
    "áéíóúýàèìòùỳâêîôûŷäëïöüÿñÁÉÍÓÚÝÀÈÌÒÙỲÂÊÎÔÛŶÄËÏÖÜŸ",
    "aeiouyaeiouyaeiouyaeiouynAEIOUYAEIOUYAEIOUYAEIOUY"
    )
    return text.translate(accents_translation_table)


def convert_example_to_SPNF(line, chineseandpunctuationextractor, max_seq_len, tokenizer):
    text_raw = line['text']
    spo_list = line['spo_list'] if 'spo_list' in line.keys() else None

    ##******************************text process*************************************
    sub_text = list() # 放置中文字符
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

    tokens = list()
    for token in sub_text:
        sub_tokens = [char for char in token]
        for sub_token in sub_tokens:
            tokens.append(sub_token)
            if len(tokens) >= max_seq_len - 2:
                break
        else:
            continue
        break

    seq_len = len(tokens)
    if seq_len > max_seq_len - 2:
        tokens = tokens[:max_seq_len - 2]
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]

    ##******************************spo process*************************************
    complex_predicate = ["上映时间", "饰演", "获奖", "配音", "票房"]
    spo_list_processed = list()
    if spo_list is not None:
        for spo in spo_list:
            # simple object既包括复杂的，又包括简单的
            if len(spo['object']) == 1:
                predicate_ = spo['predicate']
                if predicate_ in complex_predicate:
                    spo_list_processed.append({
                        "predicate": predicate_ + "_@value",
                        "subject": spo['subject'],
                        "object": spo['object']['@value']
                    })
                else:
                    spo_list_processed.append({
                        "predicate": predicate_,
                        "subject": spo["subject"],
                        "object": spo['object']['@value']
                    })
            else:
                for key, val in spo['object'].items():
                    spo_list_processed.append({
                        "predicate": spo['predicate'] + "_" + key,
                        "subject": spo['subject'],
                        "object": val
                    })
    return {"tokens": tokens, "spo_list": spo_list_processed} if spo_list is not None else {"tokens": tokens}


def data_process(input_doc, relational_alphabet, tokenizer, max_seq_len):
    samples = []
    chineseandpunctuationextractor = ChineseAndPunctuationExtractor()

    with open(input_doc) as f:
        lines = f.readlines()
    for i, line in enumerate(tqdm(lines)):
        line = json.loads(line)
        line = convert_example_to_SPNF(line, chineseandpunctuationextractor, max_seq_len, tokenizer)
        token_sent = line['tokens']
        sent_id = tokenizer.convert_tokens_to_ids(token_sent)
        if 'spo_list' in line:
            triples = line['spo_list']
            target = {"relation": [], "head_start_index": [], "head_end_index": [], "tail_start_index": [], "tail_end_index": []}
            for triple in triples:
                head_entity = triple["subject"] # 主体
                tail_entity = triple["object"] # 客体
                head_token = [char for char in head_entity] # 主体token
                tail_token = [char for char in tail_entity]

                relation_id = relational_alphabet.get_index(triple["predicate"]) # 关系id
                head_start_index, head_end_index = list_index(head_token, token_sent) # 头实体开头和结尾的id
                assert head_end_index >= head_start_index
                tail_start_index, tail_end_index = list_index(tail_token, token_sent)
                assert tail_end_index >= tail_start_index
                target["relation"].append(relation_id)
                target["head_start_index"].append(head_start_index)
                target["head_end_index"].append(head_end_index)
                target["tail_start_index"].append(tail_start_index)
                target["tail_end_index"].append(tail_end_index)
            if target["relation"] == []:
                continue
            samples.append([i, sent_id, target])
        else:
            samples.append([i, sent_id])
    return samples


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def generate_span(start_logits, end_logits, info, args):
    seq_lens = info["seq_len"] # including [CLS] and [SEP]
    sent_idxes = info["sent_idx"]
    _Prediction = collections.namedtuple(
        "Prediction", ["start_index", "end_index", "start_prob", "end_prob"]
    )
    output = {}
    start_probs = start_logits.softmax(-1) # (8 x 10 x length)
    end_probs = end_logits.softmax(-1)
    start_probs = start_probs.cpu().tolist()
    end_probs = end_probs.cpu().tolist()
    for (start_prob, end_prob, seq_len, sent_idx) in zip(start_probs, end_probs, seq_lens, sent_idxes):
        # 拿出每个句子-shape: (10 x length)
        output[sent_idx] = {}
        for triple_id in range(args.num_generated_triples):
            # 拿出每个三元组
            predictions = []
            start_indexes = _get_best_indexes(start_prob[triple_id], args.n_best_size)
            end_indexes = _get_best_indexes(end_prob[triple_id], args.n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the sentence. We throw out all
                    # invalid predictions.
                    if start_index >= (seq_len-1): # [SEP]
                        continue
                    if end_index >= (seq_len-1):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > args.max_span_length:
                        continue
                    predictions.append(
                        _Prediction(
                            start_index=start_index,
                            end_index=end_index,
                            start_prob=start_prob[triple_id][start_index],
                            end_prob=end_prob[triple_id][end_index],
                        )
                    )
            output[sent_idx][triple_id] = predictions
    return output # shape: batch x 10 x len(predictions)


def generate_relation(pred_rel_logits, info, args):
    # pred_rel_logits: (batch x 10 x 172)
    rel_probs, pred_rels = torch.max(pred_rel_logits.softmax(-1), dim=2)
    rel_probs = rel_probs.cpu().tolist()
    pred_rels = pred_rels.cpu().tolist()
    sent_idxes = info["sent_idx"]
    output = {}
    _Prediction = collections.namedtuple(
        "Prediction", ["pred_rel", "rel_prob"]
    )
    for (rel_prob, pred_rel, sent_idx) in zip(rel_probs, pred_rels, sent_idxes):
        # 拿出每个句子-shape: (10 x 172)
        output[sent_idx] = {}
        for triple_id in range(args.num_generated_triples):
            # 拿出10种三元组下的每个预测的关系
            output[sent_idx][triple_id] = _Prediction(
                            pred_rel=pred_rel[triple_id], # 预测的关系id
                            rel_prob=rel_prob[triple_id]) # 预测的关系概率
    return output # shape: batch x 10


def generate_triple(output, info, args, num_classes):
    _Pred_Triple = collections.namedtuple(
        "Pred_Triple", ["pred_rel", "rel_prob", "head_start_index", "head_end_index", "head_start_prob", "head_end_prob", "tail_start_index", "tail_end_index", "tail_start_prob", "tail_end_prob"]
    )
    # pred_head_ent_dict shape: (batch * num_generated_triples * batch_seq_max_length)-(8 x 10 x length)
    pred_head_ent_dict = generate_span(output["head_start_logits"], output["head_end_logits"], info, args)
    pred_tail_ent_dict = generate_span(output["tail_start_logits"], output["tail_end_logits"], info, args)
    pred_rel_dict = generate_relation(output['pred_rel_logits'], info, args)
    triples = {}
    for sent_idx in pred_rel_dict:
        triples[sent_idx] = []
        for triple_id in range(args.num_generated_triples): # 共10种情况
            pred_rel = pred_rel_dict[sent_idx][triple_id]
            pred_head = pred_head_ent_dict[sent_idx][triple_id] # 包含多个实体对
            pred_tail = pred_tail_ent_dict[sent_idx][triple_id]
            triple = generate_strategy(pred_rel, pred_head, pred_tail, num_classes, _Pred_Triple)
            if triple:
                triples[sent_idx].append(triple) # 放着10对三元组
    # print(triples)
    return triples # {0: [{"pred_rel": int, "rel_prob": float, ...}, {}, ...省略10个字典], 1: []}


def generate_strategy(pred_rel, pred_head, pred_tail, num_classes, _Pred_Triple):
    if pred_rel.pred_rel != num_classes:
        if pred_head and pred_tail:
            for ele in pred_head:
                if ele.start_index != 0:
                    break
            head = ele
            for ele in pred_tail:
                if ele.start_index != 0:
                    break
            tail = ele
            return _Pred_Triple(pred_rel=pred_rel.pred_rel, rel_prob=pred_rel.rel_prob, head_start_index=head.start_index, head_end_index=head.end_index, head_start_prob=head.start_prob, head_end_prob=head.end_prob, tail_start_index=tail.start_index, tail_end_index=tail.end_index, tail_start_prob=tail.start_prob, tail_end_prob=tail.end_prob)
        else:
            return
    else:
        return


# def strict_strategy(pred_rel, pred_head, pred_tail, num_classes, _Pred_Triple):
#     if pred_rel.pred_rel != num_classes:
#         if pred_head and pred_tail:
#             if pred_head[0].start_index != 0 and pred_tail[0].start_index != 0:
#                 return _Pred_Triple(pred_rel=pred_rel.pred_rel, rel_prob=pred_rel.rel_prob, head_start_index=pred_head[0].start_index, head_end_index=pred_head[0].end_index, head_start_prob=pred_head[0].start_prob, head_end_prob=pred_head[0].end_prob, tail_start_index=pred_tail[0].start_index, tail_end_index=pred_tail[0].end_index, tail_start_prob=pred_tail[0].start_prob, tail_end_prob=pred_tail[0].end_prob)
#             else:
#                 return
#         else:
#             return
#     else:
#         return


def formulate_gold(target, info):
    # target: [{"relation": [], "head_start_index": [], "head_end_index": [], "tail_start_index": [], "tail_end_index": []}, ...]
    sent_idxes = info["sent_idx"]
    gold = {}
    for i in range(len(sent_idxes)):
        gold[sent_idxes[i]] = []
        for j in range(len(target[i]["relation"])):
            gold[sent_idxes[i]].append(
                (target[i]["relation"][j].item(), target[i]["head_start_index"][j].item(), target[i]["head_end_index"][j].item(), target[i]["tail_start_index"][j].item(), target[i]["tail_end_index"][j].item())
            )
    return gold # {0: [(三元组1), (三元组2), ...], 1: [...]}



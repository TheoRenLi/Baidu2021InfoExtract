"""duee 1.0 data predict post-process"""

import os
import sys
import json
import argparse

from utils import read_by_lines, write_by_lines, extract_result


def predict_data_process(trigger_file, role_file, schema_file, save_path):
    """predict_data_process"""
    pred_ret = []
    trigger_datas = read_by_lines(trigger_file)
    role_data = read_by_lines(role_file)
    schema_datas = read_by_lines(schema_file)
    print("trigger predict {} load from {}".format(
        len(trigger_datas), trigger_file))
    print("role predict {} load from {}".format(len(role_data), role_file))
    print("schema {} load from {}".format(len(schema_datas), schema_file))

    schema = {}
    for s in schema_datas:
        d_json = json.loads(s)
        schema[d_json["event_type"]] = [r["role"] for r in d_json["role_list"]] # 字典：key-event_type，value-role_list， {event_type: [role1, role2, ...]}

    # process the role data
    sent_role_mapping = {}
    for d in role_data:
        d_json = json.loads(d)
        r_ret = extract_result(d_json["text"], d_json["pred"]["labels"])
        role_ret = {}
        for r in r_ret: # 对每个预测到的论元组，即字典{'start': , 'text': , 'type': }
            role_type = r["type"] # 获取预测的role
            if role_type not in role_ret:
                role_ret[role_type] = []
            role_ret[role_type].append("".join(r["text"])) # 汇集论元对应的实例化的词，即{'role_type': [role_arg1, role_arg2, ...]}
        sent_role_mapping[d_json["id"]] = role_ret # 当前句子的{id : {'role_type': [role_arg1, role_arg2, ...]}}

    for d in trigger_datas:
        d_json = json.loads(d)
        t_ret = extract_result(d_json["text"], d_json["pred"]["labels"]) # [{'start': , 'text': , 'type': }, ...]
        pred_event_types = list(set([t["type"] for t in t_ret])) # 获取预测的trigger类型，即事件类型。利用触发词来拿到事件类型，一个句子可以有多个事件类型，所以用了序列标注的方法
        event_list = []
        for event_type in pred_event_types:
            role_list = schema[event_type]
            arguments = []
            for role_type, ags in sent_role_mapping[d_json["id"]].items():
                if role_type not in role_list: # 这里的判断可以排除理应不在本事件类型中出现的论元；我们只关心本事件类型出现的论元是否被预测到
                    continue
                for arg in ags:
                    if len(arg) == 1:
                        continue
                    arguments.append({"role": role_type, "argument": arg})
            event = {"event_type": event_type, "arguments": arguments} 
            event_list.append(event)
        pred_ret.append({
            "id": d_json["id"],
            "text": d_json["text"],
            "event_list": event_list
        })
    pred_ret = [json.dumps(r, ensure_ascii=False) for r in pred_ret] # 最后的预测数据里的event_list是没有index的，只有event_type, {role, argument}
    print("submit data {} save to {}".format(len(pred_ret), save_path))
    write_by_lines(save_path, pred_ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Official evaluation script for DuEE version 1.0")
    parser.add_argument(
        "--trigger_file", help="trigger model predict data path", required=True)
    parser.add_argument(
        "--role_file", help="role model predict data path", required=True)
    parser.add_argument("--schema_file", help="schema file path", required=True)
    parser.add_argument("--save_path", help="save file path", required=True)
    args = parser.parse_args()
    predict_data_process(args.trigger_file, args.role_file, args.schema_file,
                         args.save_path)

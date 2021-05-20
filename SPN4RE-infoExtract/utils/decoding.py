import json
import codecs
import zipfile
from transformers import BertTokenizer
from utils.functions import convert_example_to_SPNF
from utils.extract_chinese_and_punct import ChineseAndPunctuationExtractor

def is_complex_rel_in_spo_list(complex_rel, subject, spo_list):
    for i, spo in enumerate(spo_list):
        if complex_rel == spo['predicate'] and subject == spo['subject']:
            return i
    return None

def find_object_type(suffix, obj_type_list):
    true_type = None
    for type_ in obj_type_list:
        if suffix in type_:
            true_type = type_.split('_')[0]
    return true_type

def delete_overlap(spo_list):
    res = list()
    for spo in spo_list:
        if spo not in res:
            res.append(spo)
    return res

def delete_no_atValue(spo_list):
    res = list()
    for spo in spo_list:
        if '@value' in spo['object']:
            res.append(spo)
    return res

def decoding(args, predictions, pred2id):
    """
    Model's output: batch sentence triples -> formatted output
    """
    tokenizer = BertTokenizer.from_pretrained(args.bert_directory, cache_dir=args.cache_dir)
    # tokenizer = BertTokenizer.from_pretrained(args.bert_directory)
    chineseandpunctuationextractor = ChineseAndPunctuationExtractor()

    with open(args.pred2sub_obj, 'r', encoding='utf-8') as f: # predicate -> subject_type and object_type
        p2so = json.load(f) 
    with open(args.test_file, 'r', encoding='utf-8') as f: # test file's sentences
        sentences = [json.loads(line) for line in f.readlines()]
    id2pred = {val: key for key, val in pred2id.items()} # id -> predicate
    
    formatted_outputs = list()
    for sent_idx in predictions:
        sent_raw = sentences[sent_idx]['text']
        sent_token = convert_example_to_SPNF(sentences[sent_idx], chineseandpunctuationextractor, args.max_seq_len, tokenizer)['tokens']
        triples = list(set([(ele.pred_rel, ele.head_start_index, ele.head_end_index, ele.tail_start_index, ele.tail_end_index) for ele in predictions[sent_idx]]))
        
        formatted_instance = dict()
        spo_list = list()
        for triple in triples:
            relation = id2pred[triple[0]] if triple[0] in id2pred else None
            if relation is not None:
                head = "".join(sent_token[triple[1] : triple[2] + 1])
                tail = "".join(sent_token[triple[3] : triple[4] + 1])
                if "_" in relation: # complex object
                    rel, suffix = relation.split('_')
                    idx_spo = is_complex_rel_in_spo_list(rel, head, spo_list)
                    if idx_spo is not None: # has existed
                        obj_type = find_object_type(suffix, p2so[rel]['object_type'])
                        if suffix in spo_list[idx_spo]['object']: # has the same suffix
                            spo_list.append({
                                'predicate': rel,
                                'object_type': {suffix: obj_type},
                                'subject_type': p2so[rel]['subject_type'],
                                'object': {suffix: tail},
                                'subject': head
                            })
                        else: # has different suffix
                            spo_list[idx_spo]['object'].update({suffix: tail})
                            spo_list[idx_spo]['object_type'].update({suffix: obj_type})
                    else: # non existed
                        obj_type = find_object_type(suffix, p2so[rel]['object_type'])
                        spo_list.append({
                            'predicate': rel,
                            'object_type': {suffix: obj_type},
                            'subject_type': p2so[rel]['subject_type'],
                            'object': {suffix: tail},
                            'subject': head
                        })
                else: # simple object
                    spo_list.append({
                        'predicate': relation,
                        'object_type': {'@value': p2so[relation]['object_type'][0]},
                        'subject_type': p2so[relation]['subject_type'],
                        'object': {'@value': tail},
                        'subject': head
                    })
        spo_list = delete_overlap(spo_list) # delete overlapped triples
        spo_list = delete_no_atValue(spo_list) # delete on @value triple
        formatted_instance['text'] = sent_raw
        formatted_instance['spo_list'] = spo_list
        formatted_outputs.append(formatted_instance)

    return formatted_outputs


def write_prediction_results(formatted_outputs, file_path):
    """write the prediction results"""

    with codecs.open(file_path, 'w', 'utf-8') as f:
        for formatted_instance in formatted_outputs:
            json_str = json.dumps(formatted_instance, ensure_ascii=False)
            f.write(json_str)
            f.write('\n')
        zipfile_path = file_path + '.zip'
        f = zipfile.ZipFile(zipfile_path, 'w', zipfile.ZIP_DEFLATED)
        f.write(file_path)

    return zipfile_path

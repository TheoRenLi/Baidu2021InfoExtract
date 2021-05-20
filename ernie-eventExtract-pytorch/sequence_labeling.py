"""
sequence labeling
"""
import ast
import os
import json
import warnings
import random
import argparse
import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from transformers.optimization import Adafactor
from transformers import AutoModel, AutoTokenizer

from stack_pad_tuple import Stack, Tuple, Pad
from chunk import ChunkEvaluator

from utils import read_by_lines, write_by_lines, load_dict

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=20, help="Number of epoches for fine-tuning.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--lr_decoder", type=float, default=1e-3, help="learning rate used to decoder.")

parser.add_argument("--tag_path", type=str, default=None, help="tag set path")
parser.add_argument("--train_data", type=str, default=None, help="train data")
parser.add_argument("--dev_data", type=str, default=None, help="dev data")
parser.add_argument("--test_data", type=str, default=None, help="test data")
parser.add_argument("--predict_data", type=str, default=None, help="predict data")

parser.add_argument("--do_train", type=ast.literal_eval, default=True, help="do train")
parser.add_argument("--do_predict", type=ast.literal_eval, default=False, help="do predict")

parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--max_seq_len", type=int, default=300, help="Number of words of the longest seqence.")
parser.add_argument("--valid_step", type=int, default=200, help="validation step")
parser.add_argument("--skip_step", type=int, default=100, help="skip step")
parser.add_argument("--batch_size", type=int, default=16, help="Total examples' number in batch for training.")
parser.add_argument("--checkpoints", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--init_ckpt", type=str, default=None, help="already pretraining model checkpoint")
parser.add_argument("--predict_save_path", type=str, default=None, help="predict data save path")
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'cuda'], default="cpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--num_head", type=int, default=2, help="numbers of head for attention")
parser.add_argument("--out_size", type=int, default=61, help="middel size of attention")
args = parser.parse_args()

def set_seed(args):
    """sets random seed"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def log(mode):
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.DEBUG)
    handler = logging.FileHandler(mode + "_log.txt", mode="w")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


def to_var(x, device="cuda"):
    if device == "cuda":
        return Variable(torch.from_numpy(x).cuda())
    else:
        return Variable(torch.from_numpy(x))


class ErnieForTokenClassification(nn.Module):
    def __init__(self, num_classes=2, dropout=0.2):
        super(ErnieForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.ernie = AutoModel.from_pretrained("nghuyong/ernie-1.0")
        self.dropout = nn.Dropout(dropout if dropout is not None else self.ernie.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.ernie.config.hidden_size, num_classes)
        self.init_weights()

    def init_weights(self):
        self.ernie.embeddings.word_embeddings.weight.requires_grad = False
        self.ernie.embeddings.token_type_embeddings.weight.requires_grad = False
        self.ernie.embeddings.position_embeddings.weight.requires_grad = False

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        output = self.ernie(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask)
        sequence_output = output.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


@torch.no_grad()
def evaluate(model, criterion, metric, num_label, data_loader, device):
    """evaluate"""
    model.eval()
    metric.reset()
    losses = []
    for input_ids, seg_ids, seq_lens, labels in data_loader:
        input_ids = to_var(input_ids, device=device).long()
        seg_ids = to_var(seg_ids, device=device).long()
        labels = to_var(labels, device=device)

        logits = model(input_ids, seg_ids)
        preds = torch.argmax(logits, dim=-1)
        loss = torch.mean(criterion(logits.reshape([-1, num_label]), labels.reshape([-1])))
        if device == "cuda":
            losses.append(loss.cpu().numpy())
            preds = preds.cpu()
            labels = labels.cpu()
        else:
            losses.append(loss.numpy())
        
        n_infer, n_label, n_correct = metric.compute(None, seq_lens, preds, labels)
        metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
        precision, recall, f1_score = metric.accumulate()
    avg_loss = np.mean(losses)
    model.train()

    return precision, recall, f1_score, avg_loss


def convert_example_to_feature(example, tokenizer, label_vocab=None, max_seq_len=512, no_entity_label="O", ignore_label=-1, is_test=False):
    tokens, labels = example
    tokenized_input_1 = tokenizer(
        tokens,
        is_split_into_words=True,
        return_length=True,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len)

    tokenized_input_2 = tokenizer(
        tokens,
        is_split_into_words=True,
        return_length=True,
        truncation=True,
        max_length=max_seq_len)

    input_ids = tokenized_input_1['input_ids']
    token_type_ids = tokenized_input_1['token_type_ids']
    seq_len = tokenized_input_2['length']

    if is_test:
        return input_ids, token_type_ids, seq_len
    elif label_vocab is not None:
        labels = labels[:(max_seq_len-2)]
        encoded_label = [no_entity_label] + labels + [no_entity_label]
        encoded_label = [label_vocab[x] for x in encoded_label]
        # padding label 
        encoded_label += [ignore_label]* (max_seq_len - len(encoded_label))
        return input_ids, token_type_ids, seq_len, encoded_label


class DuEventExtraction(Dataset):
    """DuEventExtraction"""
    def __init__(self, data_path, tag_path):
        self.label_vocab = load_dict(tag_path)
        self.word_ids = []
        self.label_ids = []
        with open(data_path, 'r', encoding='utf-8') as fp:
            # skip the head line
            next(fp)
            for line in fp.readlines():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                self.word_ids.append(words)
                self.label_ids.append(labels)
        self.label_num = max(self.label_vocab.values()) + 1

    def __len__(self):
        return len(self.word_ids)

    def __getitem__(self, index):
        return self.word_ids[index], self.label_ids[index]


def do_train():
    device= args.device
    logger = log(mode="train")
    logger.info("current device is {}".format(device))
    set_seed(args)

    no_entity_label = "O"
    ignore_label = -1

    logger.info("Loading the ernie model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
    label_map = load_dict(args.tag_path)
    id2label = {val: key for key, val in label_map.items()}
    model = ErnieForTokenClassification(num_classes=len(label_map))

    device_ids = [0, 1]
    model = nn.DataParallel(model, device_ids=device_ids)
    model.to(torch.device(args.device))

    logger.info("\n============start train==========")
    train_ds = DuEventExtraction(args.train_data, args.tag_path)
    dev_ds = DuEventExtraction(args.dev_data, args.tag_path)
    test_ds = DuEventExtraction(args.test_data, args.tag_path)

    trans_func = partial(
        convert_example_to_feature,
        tokenizer=tokenizer,
        label_vocab=train_ds.label_vocab,
        max_seq_len=args.max_seq_len,
        no_entity_label=no_entity_label,
        ignore_label=ignore_label,
        is_test=False)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]), # input ids
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]), # token type ids
        Stack(), # sequence lens
        Pad(axis=0, pad_val=ignore_label) # labels
    ): fn(list(map(trans_func, samples)))

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=batchify_fn)
    dev_loader = DataLoader(
        dataset=dev_ds,
        batch_size=args.batch_size,
        collate_fn=batchify_fn)
    # test_loader = DataLoader(
    #     dataset=test_ds,
    #     batch_size=args.batch_size,
    #     collate_fn=batchify_fn)

    num_training_steps = len(train_loader) * args.num_epoch

    optimizer = optim.AdamW(params=model.parameters(), lr=args.learning_rate)
    optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

    metric = ChunkEvaluator(label_list=train_ds.label_vocab.keys(), suffix=True)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)

    step, best_f1 = 0, 0.0
    model.train()
    for epoch in range(args.num_epoch):
        for idx, (input_ids, token_type_ids, seq_lens, labels) in enumerate(train_loader):
            input_ids = to_var(input_ids, device=device).long()
            token_type_ids = to_var(token_type_ids, device=device).long()
            labels = to_var(labels, device=device)

            logits = model(input_ids, token_type_ids).reshape([-1, train_ds.label_num])
            loss = torch.mean(criterion(logits, labels.reshape([-1])))
            optimizer.module.zero_grad()
            loss.backward()
            optimizer.module.step()
            if device == "cuda":
                loss_item = loss.detach().cpu().numpy().item()
            else:
                loss_item = loss.detach().numpy().item()
            if step > 0 and step % args.skip_step == 0:
                logger.info(f'train epoch: {epoch} - step: {step} (total: {num_training_steps}) - loss: {loss_item:.6f}')
            if step > 0 and step % args.valid_step == 0:
                p, r, f1, avg_loss = evaluate(model, criterion, metric, len(label_map), dev_loader, device)
                logger.info(f'dev step: {step} - loss: {avg_loss:.5f}, precision: {p:.5f}, recall: {r:.5f}, ' \
                        f'f1: {f1:.5f} current best {best_f1:.5f}')
                if f1 > best_f1:
                    best_f1 = f1
                    logger.info(f'==============================================save best model ' \
                            f'best performerence {best_f1:5f}')
                    if not os.path.exists(args.checkpoints):
                        os.mkdir(args.checkpoints)
                    torch.save(model.module.state_dict(), '{}/best.pt'.format(args.checkpoints))
            step += 1

    # save the final model
    torch.save(model.module.state_dict(), '{}/final.pt'.format(args.checkpoints))


def do_predict():
    device= args.device

    no_entity_label = "O"
    ignore_label = -1

    tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
    label_map = load_dict(args.tag_path)
    id2label = {val: key for key, val in label_map.items()}
    model = ErnieForTokenClassification(num_classes=len(label_map))

    device_ids = [0, 1]
    model = nn.DataParallel(model, device_ids=device_ids)
    model.to(torch.device(device))

    print("============start predict==========")
    if not args.init_ckpt or not os.path.isfile(args.init_ckpt):
        raise Exception("init checkpoints {} not exist".format(args.init_ckpt))
    else:
        state_dict = torch.load(args.init_ckpt)
        model.load_state_dict(state_dict)
        model = model.module
        print("Loaded parameters from %s" % args.init_ckpt)

    # load data from predict file
    sentences = read_by_lines(args.predict_data) # origin data format
    sentences = [json.loads(sent) for sent in sentences]

    encoded_inputs_list = []
    for sent in sentences:
        sent = sent["text"].replace(" ", "\002")
        input_ids, token_type_ids, seq_len = convert_example_to_feature([list(sent), []], tokenizer,
                    max_seq_len=args.max_seq_len, is_test=True)
        encoded_inputs_list.append((input_ids, token_type_ids, seq_len))

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]), # input_ids
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]), # token_type_ids
        Stack() # sequence lens
    ): fn(samples)
    # Seperates data into some batches.
    batch_encoded_inputs = [encoded_inputs_list[i: i + args.batch_size]
                            for i in range(0, len(encoded_inputs_list), args.batch_size)]
    results = []
    model.eval()
    for batch in batch_encoded_inputs:
        input_ids, token_type_ids, seq_lens = batchify_fn(batch)
        input_ids = to_var(input_ids, device=device).long()
        token_type_ids = to_var(token_type_ids, device=device).long()

        logits = model(input_ids, token_type_ids)
        probs = torch.softmax(logits, dim=-1)
        if device == "cuda":
            probs_ids = torch.argmax(probs, -1).detach().cpu().numpy()
            probs = probs.detach().cpu().numpy()
        else:
            probs_ids = torch.argmax(probs, -1).detach().numpy()
            probs = probs.detach().numpy()
        for p_list, p_ids, seq_len in zip(probs.tolist(), probs_ids.tolist(), seq_lens.tolist()):
            prob_one = [p_list[index][pid] for index, pid in enumerate(p_ids[1: seq_len[0] - 1])]
            label_one = [id2label[pid] for pid in p_ids[1: seq_len[0] - 1]]
            results.append({"probs": prob_one, "labels": label_one})
    assert len(results) == len(sentences)
    for sent, ret in zip(sentences, results):
        sent["pred"] = ret
    sentences = [json.dumps(sent, ensure_ascii=False) for sent in sentences]
    write_by_lines(args.predict_save_path, sentences)
    print("save data {} to {}".format(len(sentences), args.predict_save_path))



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

    if args.do_train:
        do_train()
    elif args.do_predict:
        do_predict()

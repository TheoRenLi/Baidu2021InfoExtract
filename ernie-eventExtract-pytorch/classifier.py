"""
classification
"""
import csv
import ast
import os
import json
import traceback
import warnings
import random
import argparse
from collections import namedtuple
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from transformers.optimization import Adafactor
from transformers import AutoModel, AutoTokenizer

import metrics
from stack_pad_tuple import Stack, Tuple, Pad

from utils import read_by_lines, write_by_lines, load_dict

# warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--tag_path", type=str, default=None, help="tag set path")
parser.add_argument("--train_data", type=str, default=None, help="train data")
parser.add_argument("--dev_data", type=str, default=None, help="dev data")
parser.add_argument("--test_data", type=str, default=None, help="test data")
parser.add_argument("--predict_data", type=str, default=None, help="predict data")
parser.add_argument("--do_train", type=ast.literal_eval, default=True, help="do train")
parser.add_argument("--do_predict", type=ast.literal_eval, default=True, help="do predict")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion params for warmup strategy")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--valid_step", type=int, default=100, help="validation step")
parser.add_argument("--skip_step", type=int, default=20, help="skip step")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--checkpoints", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--init_ckpt", type=str, default=None, help="already pretraining model checkpoint")
parser.add_argument("--predict_save_path", type=str, default=None, help="predict data save path")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'cuda'], default="cpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()


class ErnieForSequenceClassification(nn.Module):
    def __init__(self, num_classes, dropout=None):
        super(ErnieForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.ernie = AutoModel.from_pretrained("nghuyong/ernie-1.0")
        self.dropout = nn.Dropout(dropout if dropout is not None else self.ernie.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.ernie.config.hidden_size, num_classes)
        
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.classifier.weight, mean=0.0, std=self.ernie.config.initializer_range)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        output = self.ernie(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask)
        pooler_output = self.dropout(output.pooler_output)
        logits = self.classifier(pooler_output)
        return logits


def set_seed(random_seed):
    """sets random seed"""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def to_var(x, device="cuda"):
    if device == "cuda":
        return Variable(torch.from_numpy(x).cuda())
    else:
        return Variable(torch.from_numpy(x))


def convert_example(example, tokenizer, label_map=None, max_seq_len=512, is_test=False):
    """convert_example"""
    has_text_b = False
    if isinstance(example, dict):
        has_text_b = "text_b" in example.keys()
    else:
        has_text_b = "text_b" in example._fields

    text_b = None
    if has_text_b:
        text_b = example.text_b

    tokenized_input = tokenizer(
        text=example.text_a,
        text_pair=text_b,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len)

    input_ids = tokenized_input['input_ids']
    token_type_ids = tokenized_input['token_type_ids']


    if is_test:
        return input_ids, token_type_ids
    else:
        label = np.array([label_map[example.label]], dtype=np.int64)
        return input_ids, token_type_ids, label


class DuEventExtraction(Dataset):
    """Du"""
    def __init__(self, data_path, tag_path):
        self.label_vocab = load_dict(tag_path) 
        self.examples = self._read_tsv(data_path)

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            headers = next(reader)
            text_indices = [
                index for index, h in enumerate(headers) if h != "label"]
            Example = namedtuple('Example', headers)
            examples = []
            for line in reader:
                for index, text in enumerate(line):
                    if index in text_indices:
                        line[index] = text
                try:
                    example = Example(*line)
                except Exception as e:
                    traceback.print_exc()
                    raise Exception(e)
                examples.append(example)
            return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def data_2_examples(datas):
    """data_2_examples"""
    has_text_b, examples = False, []
    if isinstance(datas[0], list):
        Example = namedtuple('Example', ["text_a", "text_b"])
        has_text_b = True
    else:
        Example = namedtuple('Example', ["text_a"])
    for item in datas:
        if has_text_b:
            example = Example(text_a=item[0], text_b=item[1])
        else:
            example = Example(text_a=item)
        examples.append(example)
    return examples


@torch.no_grad()
def evaluate(model, criterion, metric, data_loader, device):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`torch.nn.Module`): A model to classify texts.
        data_loader(obj:`DataLoader`): The dataset loader which generates batches.
        criterion(obj:`torch.nn.Module`): It can compute the loss.
        metric(obj:`metrics.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        input_ids = to_var(input_ids, device=device).long()
        token_type_ids = to_var(token_type_ids, device=device).long()
        labels = to_var(labels, device=device).squeeze()

        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)

        if device == "cuda":
            loss = loss.detach().cpu()
            logits = logits.detach().cpu()
            labels = labels.detach().cpu()
        
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accuracy = metric.accumulate()
    metric.reset()
    model.train()
    return float(np.mean(losses)), accuracy



def do_train():
    device = args.device
    set_seed(args.seed)

    label_map = load_dict(args.tag_path)
    id2label = {val: key for key, val in label_map.items()}

    print("Loading the ernie model and tokenizer...")
    model = ErnieForSequenceClassification(num_classes=len(label_map))
    model.to(torch.device(device))
    tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")

    print("============start train==========")
    train_ds = DuEventExtraction(args.train_data, args.tag_path)
    dev_ds = DuEventExtraction(args.dev_data, args.tag_path)
    # test_ds = DuEventExtraction(args.test_data, args.tag_path)

    trans_func = partial(
        convert_example, tokenizer=tokenizer, label_map=label_map, max_seq_len=args.max_seq_len)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),
        Stack(dtype="int64")  # label
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
    metric = metrics.Accuracy()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(lr=args.learning_rate, params=model.parameters())

    best_performerence = 0.0
    model.train()
    for epoch in range(args.num_epoch):
        for step, (input_ids, token_type_ids, labels) in enumerate(train_loader):
            input_ids = to_var(input_ids, device=device).long()
            token_type_ids = to_var(token_type_ids, device=device).long()
            labels = to_var(labels, device=device).squeeze()

            logits = model(input_ids, token_type_ids)
            # print("labels shape: ", labels.shape, "logits shape: ", logits.shape)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)

            if device == "cuda":
                probs = probs.detach().cpu()
                labels = labels.detach().cpu()
            # calculate accuracy
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if device == "cuda":
                loss_item = loss.detach().cpu().numpy().item()
            else:
                loss_item = loss.detach().numpy().item()
            if step > 0 and step % args.skip_step == 0:
                print(f'train epoch: {epoch} - step: {step} (total: {num_training_steps}) ' \
                    f'- loss: {loss_item:.6f} acc {acc:.5f}')
            if step > 0 and step % args.valid_step == 0:
                loss_dev, acc_dev = evaluate(model, criterion, metric, dev_loader, device)
                print(f'dev step: {step} - loss: {loss_dev:.6f} accuracy: {acc_dev:.5f}, ' \
                        f'current best {best_performerence:.5f}')
                if acc_dev > best_performerence:
                    best_performerence = acc_dev
                    print(f'==============================================save best model ' \
                            f'best performerence {best_performerence:5f}')
                    if not os.path.exists(args.checkpoints):
                        os.mkdir(args.checkpoints)
                    torch.save(model.state_dict(), '{}/best.pt'.format(args.checkpoints))
            step += 1

    # save the final model
    torch.save(model.state_dict(), '{}/final.pt'.format(args.checkpoints))


def do_predict():
    set_seed(args.seed)
    device = args.device

    label_map = load_dict(args.tag_path)
    id2label = {val: key for key, val in label_map.items()}

    model = ErnieForSequenceClassification(num_classes=len(label_map))
    model.to(torch.device(device))
    tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")

    print("============start predict==========")
    if not args.init_ckpt or not os.path.isfile(args.init_ckpt):
        raise Exception("init checkpoints {} not exist".format(args.init_ckpt))
    else:
        state_dict = torch.load(args.init_ckpt)
        model.load_state_dict(state_dict)
        print("Loaded parameters from %s" % args.init_ckpt)

    # load data from predict file
    sentences = read_by_lines(args.predict_data) # origin data format
    sentences = [json.loads(sent) for sent in sentences]

    encoded_inputs_list = []
    for sent in sentences:
        sent = sent["text"]
        input_sent = [sent]  # only text_a
        if "text_b" in sent:
            input_sent = [[sent, sent["text_b"]]]  # add text_b
        example = data_2_examples(input_sent)[0]
        input_ids, token_type_ids = convert_example(example, tokenizer,
                    max_seq_len=args.max_seq_len, is_test=True)
        encoded_inputs_list.append((input_ids, token_type_ids))

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),
    ): fn(samples)
    # Seperates data into some batches.
    batch_encoded_inputs = [encoded_inputs_list[i: i + args.batch_size]
                            for i in range(0, len(encoded_inputs_list), args.batch_size)]
    results = []
    model.eval()
    for batch in batch_encoded_inputs:
        input_ids, token_type_ids = batchify_fn(batch)
        input_ids = to_var(input_ids, device=device).long()
        token_type_ids = to_var(token_type_ids, device=device).long()

        logits = model(input_ids, token_type_ids)
        probs = torch.softmax(logits, dim=1)
        if device == "cuda":
            probs_ids = torch.argmax(probs, -1).detach().cpu().numpy()
            probs = probs.detach().cpu().numpy()
        else:
            probs_ids = torch.argmax(probs, -1).detach().numpy()
            probs = probs.detach().numpy()
        for prob_one, p_id in zip(probs.tolist(), probs_ids.tolist()):
            label_probs = {}
            for idx, p in enumerate(prob_one):
                label_probs[id2label[idx]] = p
            results.append({"probs": label_probs, "label": id2label[p_id]})

    assert len(results) == len(sentences)
    for sent, ret in zip(sentences, results):
        sent["pred"] = ret
    sentences = [json.dumps(sent, ensure_ascii=False) for sent in sentences]
    write_by_lines(args.predict_save_path, sentences)
    print("save data {} to {}".format(len(sentences), args.predict_save_path))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    if args.do_train:
        do_train()
    elif args.do_predict:
        do_predict()

import argparse
import os
import random
import time
import math
import json
from functools import partial
import codecs
import zipfile
import re
from tqdm import tqdm
import sys
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel
from transformers.optimization import Adafactor

from data_loader import DuIEDataset, DataCollator
from utils import decoding, find_entity, get_precision_recall_f1, write_prediction_results


parser = argparse.ArgumentParser()
parser.add_argument("--do_train", action='store_true', default=False, help="do train")
parser.add_argument("--do_predict", action='store_true', default=True, help="do predict")
parser.add_argument("--init_checkpoint", default=None, type=str, required=False, help="Path to initialize params from")
parser.add_argument("--data_path", default="./data", type=str, required=False, help="Path to data.")
parser.add_argument("--predict_data_file", default="./data/test_data.json", type=str, required=False, help="Path to data.")
parser.add_argument("--output_dir", default="./checkpoints", type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_length", default=128, type=int,help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--seed", default=42, type=int, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'cuda'], default="cuda", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()


class BCELossForDuIE(nn.Module):
    def __init__(self, ):
        super(BCELossForDuIE, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels, mask):
        loss = self.criterion(logits, labels)
        mask = mask.float()
        loss = loss * mask.unsqueeze(-1)
        loss = torch.sum(loss.mean(dim=2), dim=1) / torch.sum(mask, dim=1)
        loss = loss.mean()
        return loss


class ErnieForTokenClassification(nn.Module):
    def __init__(self, num_classes=2, dropout=None):
        super(ErnieForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.ernie = AutoModel.from_pretrained("nghuyong/ernie-1.0")
        self.dropout = nn.Dropout(dropout if dropout is not None else self.ernie.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.ernie.config.hidden_size, num_classes)
        
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.classifier.weight, mean=0.0, std=self.ernie.config.initializer_range)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):

        output = self.ernie(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask)
        sequence_output = output.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


def set_random_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.seed(seed)

def log(mode):
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.DEBUG)
    handler = logging.FileHandler(mode + "_log.txt")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


@torch.no_grad()
def evaluate(model, criterion, data_loader, file_path, mode, logger):
    """
    mode eval:
    eval on development set and compute P/R/F1, called between training.
    mode predict:
    eval on development / test set, then write predictions to \
        predict_test.json and predict_test.json.zip \
        under args.data_path dir for later submission or evaluation.
    """
    model.eval()
    probs_all = None
    seq_len_all = None
    tok_to_orig_start_index_all = None
    tok_to_orig_end_index_all = None
    loss_all = 0
    eval_steps = 0
    logger.info("\n----------------------------------IN Evaluate func-----------------------------------\n")
    for batch in tqdm(data_loader, total=len(data_loader)):
        eval_steps += 1
        input_ids, seq_len, tok_to_orig_start_index, tok_to_orig_end_index, labels = batch

        if args.device == 'cuda':
            input_ids = input_ids.cuda()
            labels = labels.cuda()

        logits = model(input_ids=input_ids)
        mask = (input_ids != 0) & (input_ids != 1) & (input_ids != 2)
        loss = criterion(logits, labels, mask)
        loss_all += loss.detach().cpu().numpy().item()
        probs = torch.sigmoid(logits).cpu()
        if probs_all is None:
            probs_all = probs.numpy()
            seq_len_all = seq_len.numpy()
            tok_to_orig_start_index_all = tok_to_orig_start_index.numpy()
            tok_to_orig_end_index_all = tok_to_orig_end_index.numpy()
        else:
            probs_all = np.append(probs_all, probs.numpy(), axis=0)
            seq_len_all = np.append(seq_len_all, seq_len.numpy(), axis=0)
            tok_to_orig_start_index_all = np.append(
                tok_to_orig_start_index_all,
                tok_to_orig_start_index.numpy(),
                axis=0)
            tok_to_orig_end_index_all = np.append(
                tok_to_orig_end_index_all,
                tok_to_orig_end_index.numpy(),
                axis=0)
    loss_avg = loss_all / eval_steps
    logger.info("eval loss: %f" % (loss_avg))

    id2spo_path = os.path.join(os.path.dirname(file_path), "id2spo.json")
    with open(id2spo_path, 'r', encoding='utf8') as fp:
        id2spo = json.load(fp)
    formatted_outputs = decoding(file_path, id2spo, probs_all, seq_len_all,
                                 tok_to_orig_start_index_all,
                                 tok_to_orig_end_index_all)
    if mode == "predict":
        predict_file_path = os.path.join(args.data_path, 'predictions.json')
    else:
        predict_file_path = os.path.join(args.data_path, 'predict_eval.json')

    predict_zipfile_path = write_prediction_results(formatted_outputs,
                                                    predict_file_path)

    if mode == "eval":
        precision, recall, f1 = get_precision_recall_f1(file_path,
                                                        predict_zipfile_path)
        os.system('rm {} {}'.format(predict_file_path, predict_zipfile_path))
        return precision, recall, f1
    elif mode != "predict":
        logger.debug("wrong mode for eval func")
        raise Exception("wrong mode for eval func")
    logger.info("Finish evaluating.")


def do_train():
    device = torch.device(args.device)
    logger = log("train")

    # Reads label_map.
    label_map_path = os.path.join(args.data_path, "predicate2id.json")
    if not (os.path.exists(label_map_path) and os.path.isfile(label_map_path)):
        sys.exit("{} dose not exists or is not a file.".format(label_map_path))
    with open(label_map_path, 'r', encoding='utf8') as fp:
        label_map = json.load(fp)
    num_classes = (len(label_map.keys()) - 2) * 2 + 2

    # Loads pretrained model ERNIE
    logger.info("Loading the model and tokenizer...")
    try:
        model = ErnieForTokenClassification(num_classes=num_classes)
    except Exception as e:
        logger.error(e)
        raise Exception("Loading model error: ", e)
    
    tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
    logger.info("Finish loading ernie model and tokenizer.")

    criterion = BCELossForDuIE()

    # Loads dataset.
    logger.info("Loading the train and develop dataset...")
    train_dataset = DuIEDataset.from_file(
        os.path.join(args.data_path, 'train_data.json'), tokenizer,
        args.max_seq_length, True)

    collator = DataCollator()
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator)
        
    eval_file_path = os.path.join(args.data_path, 'dev_data.json')
    test_dataset = DuIEDataset.from_file(eval_file_path, tokenizer,
                                         args.max_seq_length, True)
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size = args.batch_size,
        shuffle=False,
        collate_fn=collator)
    logger.info("Finish loading dataset.")

    steps_by_epoch = len(train_data_loader)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        params=[{'params': decay_params, 'weight_decay': args.weight_decay}, {'params': no_decay_params, 'weight_decay': 0.0}], 
        lr=args.learning_rate
        )

    # Starts training.
    model.to(device)
    global_step = 0
    logging_steps = 50
    save_steps = 10000
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        logger.info("\n=====start training of %d epochs=====" % epoch)
        tic_epoch = time.time()
        model.train()
        for step, batch in enumerate(train_data_loader):
            input_ids, _, _, _, labels = batch

            if args.device == 'cuda':
                input_ids = input_ids.cuda()
                labels = labels.cuda()
            input_ids = Variable(input_ids)
            labels = Variable(labels)

            logits = model(input_ids=input_ids)
            mask = (input_ids != 0) & (input_ids != 1) & (input_ids != 2)

            loss = criterion(logits, labels, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_item = loss.detach().cpu().numpy().item()
            global_step += 1

            if global_step % logging_steps == 0:
                print(
                    "epoch: %d / %d, steps: %d / %d, loss: %f, speed: %.2f step/s"
                    % (epoch, args.num_train_epochs, step, steps_by_epoch,
                       loss_item, logging_steps / (time.time() - tic_train)))
                tic_train = time.time()

            if global_step % save_steps == 0:
                logger.info("\n=====start evaluating ckpt of %d steps=====" %
                      global_step)
                precision, recall, f1 = evaluate(
                    model, criterion, test_data_loader, eval_file_path, "eval", logger)
                logger.info("precision: %.2f\t recall: %.2f\t f1: %.2f\t" %
                      (100 * precision, 100 * recall, 100 * f1))

                logger.info("saving checkpoing model_%d.pt to %s " %
                      (global_step, args.output_dir))
                if not os.path.exists(args.output_dir):
                    os.mkdir(args.output_dir)
                torch.save(model.state_dict(),
                            os.path.join(args.output_dir,
                                         "model_%d.pt" % global_step))
                model.train()  # back to train mode

        tic_epoch = time.time() - tic_epoch
        logger.info("epoch time footprint: %d hour %d min %d sec" %
              (tic_epoch // 3600, (tic_epoch % 3600) // 60, tic_epoch % 60))

    # Does final evaluation.
    logger.info("\n=====start evaluating last ckpt of %d steps=====" %
            global_step)
    precision, recall, f1 = evaluate(model, criterion, test_data_loader,
                                        eval_file_path, "eval", logger)
    logger.info("precision: %.2f\t recall: %.2f\t f1: %.2f\t" %
            (100 * precision, 100 * recall, 100 * f1))
    torch.save(model.state_dict(),
                os.path.join(args.output_dir,
                                "model_%d.pt" % global_step))
    logger.info("\n=====training complete=====")


def do_predict():
    device = torch.device(args.device)
    logger = log("test")

    # Reads label_map.
    label_map_path = os.path.join(args.data_path, "predicate2id.json")
    if not (os.path.exists(label_map_path) and os.path.isfile(label_map_path)):
        sys.exit("{} dose not exists or is not a file.".format(label_map_path))
    with open(label_map_path, 'r', encoding='utf8') as fp:
        label_map = json.load(fp)
    num_classes = (len(label_map.keys()) - 2) * 2 + 2

    # Loads pretrained model ERNIE
    logger.info("Loading the model and tokenizer...")
    try:
        model = ErnieForTokenClassification(num_classes=num_classes)
    except Exception as e:
        logger.error(e)
        raise Exception("Loading model error: ", e)
    
    tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
    criterion = BCELossForDuIE()
    logger.info("Finish loading.")

    # Loads dataset.
    test_dataset = DuIEDataset.from_file(args.predict_data_file, tokenizer,
                                         args.max_seq_length, True)
    collator = DataCollator()
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size = args.batch_size,
        shuffle=False,
        collate_fn=collator)

    # Loads model parameters.
    if not (os.path.exists(args.init_checkpoint) and
            os.path.isfile(args.init_checkpoint)):
        sys.exit("wrong directory: init checkpoints {} not exist".format(
            args.init_checkpoint))
    state_dict = torch.load(args.init_checkpoint)
    model.load_state_dict(state_dict)

    model.to(device)
    # Does predictions.
    logger.info("\n=====start predicting=====")
    evaluate(model, criterion, test_data_loader, args.predict_data_file, "predict", logger)
    logger.info("=====predicting complete=====")


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    if args.do_train:
        do_train()
    elif args.do_predict:
        do_predict()

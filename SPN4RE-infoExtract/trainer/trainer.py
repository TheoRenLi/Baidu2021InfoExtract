import torch, random, gc, os
from torch import nn, optim
from torch.autograd import Variable
from tqdm import tqdm
from transformers import AdamW
from utils.average_meter import AverageMeter
from utils.functions import formulate_gold, generate_triple
from utils.metric import metric, num_metric, overlap_metric
from models.set_criterion import SetCriterion
from utils.decoding import decoding, write_prediction_results
from utils.dataloader import TextDataset, DataCollator
from torch.utils.data import DataLoader


class Trainer(nn.Module):
    def __init__(self, model, data, args):
        super().__init__()
        self.args = args
        #self.model = nn.DataParallel(model, device_ids=[0, 1])
        self.model = model
        self.data = data
        self.num_classes = data.relational_alphabet.size()
        self.criterion = SetCriterion(self.num_classes,  loss_weight=self.get_loss_weight(args), na_coef=args.na_rel_coef, losses=["entity", "relation"], matcher=args.matcher)
        
        # iterator
        collator = DataCollator()
        train_data = TextDataset(self.data.train_data)
        valid_data = TextDataset(self.data.valid_data)
        test_data = TextDataset(self.data.test_data)
        self.test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
        self.valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
        self.train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        component = ['encoder', 'decoder']
        grouped_params = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and component[0] in n],
                'weight_decay': args.weight_decay,
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and component[0] in n],
                'weight_decay': 0.0,
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if
                           not any(nd in n for nd in no_decay) and component[1] in n],
                'weight_decay': args.weight_decay,
                'lr': args.decoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if
                           any(nd in n for nd in no_decay) and component[1] in n],
                'weight_decay': 0.0,
                'lr': args.decoder_lr
            }
        ]
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(grouped_params)
            #self.optimizer = nn.DataParallel(self.optimizer, device_ids=[0, 1])
        elif args.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(grouped_params)
            #self.optimizer = nn.DataParallel(self.optimizer, device_ids=[0, 1])
        else:
            raise Exception("Invalid optimizer.")
        if args.use_gpu:
            self.cuda()

    def train_model(self):
        best_f1 = 0
        total = len(self.train_loader) * self.args.batch_size
        for epoch in range(self.args.max_epoch):
            # Train
            self.model.train()
            #self.model.module.zero_grad()
            self.model.zero_grad()
            self.optimizer = self.lr_decay(self.optimizer, epoch, self.args.lr_decay)
            print("=== Epoch %d train ===" % epoch, flush=True)
            avg_loss = AverageMeter()
            for batch_id, (input_ids, attention_mask, targets, _) in enumerate(self.train_loader):
                # if batch_id > 10:
                #     break
                if self.args.use_gpu:
                    input_ids = Variable(input_ids.cuda())
                    attention_mask = Variable(attention_mask.cuda())
                    targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, targets)
                loss = loss / self.args.gradient_accumulation_steps
                avg_loss.update(loss.item(), 1)
                # Optimize
                loss.backward()
                if self.args.max_grad_norm != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                if (batch_id + 1) % self.args.gradient_accumulation_steps == 0:
                    #self.optimizer.module.step()
                    #self.model.module.zero_grad()
                    self.optimizer.step()
                    self.model.zero_grad()
                if batch_id % 100 == 0 and batch_id != 0:
                    print("\tEpoch: %d; Instance: %d/%d; loss: %.4f" % (epoch, batch_id * self.args.batch_size, total, avg_loss.avg), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            if not os.path.exists(self.args.generated_param_directory):
                os.mkdir(self.args.generated_param_directory)
            if epoch > 10:
                torch.save(self.model.state_dict(), self.args.generated_param_directory + "SPN4RE_%s.model" %(str(epoch)))
            # Validation
            print("=== Epoch %d Validation ===" % epoch)
            result = self.eval_model()
            f1 = result['f1']
            if f1 > best_f1:
                print("Achieving Best Result on Test Set.", flush=True)
                if not os.path.exists(self.args.generated_param_directory):
                    os.mkdir(self.args.generated_param_directory)
                torch.save(self.model.state_dict(), self.args.generated_param_directory + "%s_%s_best.model" %("SPN4RE", self.args.dataset_name))
                best_f1 = f1
                best_result_epoch = epoch
            gc.collect()
            torch.cuda.empty_cache()
        print("Best result on test set is %f achieving at epoch %d." % (best_f1, best_result_epoch), flush=True)


    def eval_model(self):
        self.model.eval()
        prediction, gold = {}, {}
        with torch.no_grad():
            for batch_id, (input_ids, attention_mask, targets, info) in enumerate(tqdm(self.valid_loader)):
                # if batch_id > 10:
                #     break
                if self.args.use_gpu:
                    input_ids = Variable(input_ids.cuda())
                    attention_mask = Variable(attention_mask.cuda())
                    targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

                gold.update(formulate_gold(targets, info)) # {0: [(三元组1), (三元组2), ...], 1: [...]}
                gen_triples = self.gen_triples(input_ids, attention_mask, info)
                prediction.update(gen_triples) # {0: [{"pred_rel": int, "rel_prob": float, ...}, {}, ...省略10个字典], 1: []}
        # num_metric(prediction, gold)
        # overlap_metric(prediction, gold)
        return metric(prediction, gold)

    def predict_model(self):
        print("Loading the saved SPN4RE model...")
        path_model = self.args.generated_param_directory + "%s_%s_best.model" %("SPN4RE", self.args.dataset_name)
        state_dict = torch.load(path_model)
        self.load_state_dict(state_dict)

        self.model.eval()
        prediction = dict()
        with torch.no_grad():
            for batch_id, (input_ids, attention_mask, info) in enumerate(tqdm(self.test_loader)):
                if self.args.use_gpu:
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda()

                gen_triples = self.gen_triples(input_ids, attention_mask, info)
                prediction.update(gen_triples)
        # *******need yourself to write extract triples*******
        formatted_outputs = decoding(self.args, prediction, self.data.relational_alphabet.instance2index)
        # Saving the formatted outputs
        if not os.path.exists(self.args.submit_dir):
            os.mkdir(self.args.submit_dir)
        predict_file_path = os.path.join(self.args.submit_dir, "predictions.json")
        predict_zipfile_path = write_prediction_results(formatted_outputs, predict_file_path)
        print('rm {} {}'.format(predict_file_path, predict_zipfile_path))

    
    def gen_triples(self, input_ids, attention_mask, info):
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            pred_triple = generate_triple(outputs, info, self.args, self.num_classes)
        return pred_triple

    def load_state_dict(self, state_dict):
        #self.model.module.load_state_dict(state_dict)
        self.model.load_state_dict(state_dict)

    @staticmethod
    def lr_decay(optimizer, epoch, decay_rate):
        # lr = init_lr * ((1 - decay_rate) ** epoch)
        if epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * (1 - decay_rate)
        return optimizer
    
    @staticmethod
    def get_loss_weight(args):
        return {"relation": args.rel_loss_weight, "head_entity": args.head_ent_loss_weight, "tail_entity": args.tail_ent_loss_weight}

from utils.alphabet import Alphabet
import os, pickle, copy, sys, copy
from utils.functions import data_process
from transformers import BertTokenizer


class Data:
    def __init__(self):
        self.relational_alphabet = Alphabet("Relation", unkflag=False, padflag=False)
        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Relation Alphabet Size: %s" % self.relational_alphabet.size())
        print("     Train  Instance Number: %s" % (len(self.train_data)))
        print("     Valid  Instance Number: %s" % (len(self.valid_data)))
        print("     Test   Instance Number: %s" % (len(self.test_data)))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def generate_instance(self, args, data_process):
        tokenizer = BertTokenizer.from_pretrained(args.bert_directory, cache_dir=args.cache_dir)
        # tokenizer = BertTokenizer.from_pretrained(args.bert_directory)
        if "train_file" in args:
            # [[0, sent_id, {"relation": [], "head_start_index": [],"head_end_index": [],"tail_start_index": [],"tail_end_index": []}], ...] for ??_loader
            self.train_data = data_process(args.train_file, self.relational_alphabet, tokenizer, args.max_seq_len)
        if "valid_file" in args:
            self.valid_data = data_process(args.valid_file, self.relational_alphabet, tokenizer, args.max_seq_len)
        if "test_file" in args:
            self.test_data = data_process(args.test_file, self.relational_alphabet, tokenizer, args.max_seq_len)

        self.relational_alphabet.close()


def build_data(args):

    file = args.generated_data_directory + args.dataset_name + "_" + args.model_name + "_data.pickle"
    if os.path.exists(file) and not args.refresh:
        data = load_data_setting(args)
    else:
        data = Data()
        data.generate_instance(args, data_process)
        save_data_setting(data, args)
    return data


def save_data_setting(data, args):
    new_data = copy.deepcopy(data)
    data.show_data_summary()
    if not os.path.exists(args.generated_data_directory):
        os.makedirs(args.generated_data_directory)
    saved_path = args.generated_data_directory + args.dataset_name + "_" + args.model_name + "_data.pickle"
    with open(saved_path, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting is saved to file: ", saved_path)


def load_data_setting(args):

    saved_path = args.generated_data_directory + args.dataset_name + "_" + args.model_name + "_data.pickle"
    with open(saved_path, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting is loaded from file: ", saved_path)
    data.show_data_summary()
    return data


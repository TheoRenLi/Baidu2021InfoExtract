from collections import defaultdict

import numpy as np
from seqeval.metrics.sequence_labeling import get_entities
from metrics import Metric
import torch


def extract_tp_actual_correct(y_true, y_pred, suffix, *args):
    entities_true = defaultdict(set)
    entities_pred = defaultdict(set)
    for type_name, start, end in get_entities(y_true, suffix):
        entities_true[type_name].add((start, end))
    for type_name, start, end in get_entities(y_pred, suffix):
        entities_pred[type_name].add((start, end))

    target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

    tp_sum = np.array([], dtype=np.int32)
    pred_sum = np.array([], dtype=np.int32)
    true_sum = np.array([], dtype=np.int32)
    for type_name in target_names:
        entities_true_type = entities_true.get(type_name, set())
        entities_pred_type = entities_pred.get(type_name, set())
        tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
        pred_sum = np.append(pred_sum, len(entities_pred_type))
        true_sum = np.append(true_sum, len(entities_true_type))
    return pred_sum, tp_sum, true_sum


class ChunkEvaluator(Metric):
    """ChunkEvaluator computes the precision, recall and F1-score for chunk detection.
    It is often used in sequence tagging tasks, such as Named Entity Recognition(NER).

    Args:
        label_list (list): The label list.
        suffix (bool): if set True, the label ends with '-B', '-I', '-E' or '-S', else the label starts with them.
    """

    def __init__(self, label_list, suffix=False):
        super(ChunkEvaluator, self).__init__()
        self.id2label_dict = dict(enumerate(label_list))
        self.suffix = suffix
        self.num_infer_chunks = 0
        self.num_label_chunks = 0
        self.num_correct_chunks = 0

    def compute(self, inputs, lengths, predictions, labels):
        labels = labels.numpy()
        predictions = predictions.numpy()
        unpad_labels = [[self.id2label_dict[idx] for idx in indexs] for indexs in self.find_idx(labels, lengths)]
        unpad_predictions = [[self.id2label_dict.get(idx, 'O') for idx in indexs] for indexs in self.find_idx(predictions, lengths)]

        pred_sum, tp_sum, true_sum = extract_tp_actual_correct(
            unpad_labels, unpad_predictions, self.suffix)
        num_correct_chunks = torch.Tensor([tp_sum.sum()])
        num_infer_chunks = torch.Tensor([pred_sum.sum()])
        num_label_chunks = torch.Tensor([true_sum.sum()])

        return num_infer_chunks, num_label_chunks, num_correct_chunks

    def find_idx(self, var_numpy, lengths):
        res = list()
        for sent_index in range(len(lengths)):
            end = lengths[sent_index][0]
            tmp = var_numpy[sent_index].tolist()
            tmp = tmp[:end]
            res.append(tmp)
        return res

    def _is_number_or_matrix(self, var):
        def _is_number_(var):
            return isinstance(
                var, int) or isinstance(var, np.int64) or isinstance(
                    var, float) or (isinstance(var, np.ndarray) and
                                    var.shape == (1, ))

        return _is_number_(var) or isinstance(var, np.ndarray)

    def update(self, num_infer_chunks, num_label_chunks, num_correct_chunks):
        """
        This function takes (num_infer_chunks, num_label_chunks, num_correct_chunks) as input,
        to accumulate and update the corresponding status of the ChunkEvaluator object. The update method is as follows:

        .. math::
                   \\\\ \\begin{array}{l}{\\text { self. num_infer_chunks }+=\\text { num_infer_chunks }} \\\\ {\\text { self. num_Label_chunks }+=\\text { num_label_chunks }} \\\\ {\\text { self. num_correct_chunks }+=\\text { num_correct_chunks }}\\end{array} \\\\

        Args:
            num_infer_chunks(int|numpy.array): The number of chunks in Inference on the given minibatch.
            num_label_chunks(int|numpy.array): The number of chunks in Label on the given mini-batch.
            num_correct_chunks(int|float|numpy.array): The number of chunks both in Inference and Label on the
                                                  given mini-batch.
        """
        if not self._is_number_or_matrix(num_infer_chunks):
            raise ValueError(
                "The 'num_infer_chunks' must be a number(int) or a numpy ndarray."
            )
        if not self._is_number_or_matrix(num_label_chunks):
            raise ValueError(
                "The 'num_label_chunks' must be a number(int, float) or a numpy ndarray."
            )
        if not self._is_number_or_matrix(num_correct_chunks):
            raise ValueError(
                "The 'num_correct_chunks' must be a number(int, float) or a numpy ndarray."
            )
        self.num_infer_chunks += num_infer_chunks
        self.num_label_chunks += num_label_chunks
        self.num_correct_chunks += num_correct_chunks

    def accumulate(self):
        """
        This function returns the mean precision, recall and f1 score for all accumulated minibatches.

        Returns:
            float: mean precision, recall and f1 score.
        """
        precision = float(
            self.num_correct_chunks /
            self.num_infer_chunks) if self.num_infer_chunks else 0.
        recall = float(self.num_correct_chunks /
                       self.num_label_chunks) if self.num_label_chunks else 0.
        f1_score = float(2 * precision * recall / (
            precision + recall)) if self.num_correct_chunks else 0.
        return precision, recall, f1_score

    def reset(self):
        """
        Reset function empties the evaluation memory for previous mini-batches.
        """
        self.num_infer_chunks = 0
        self.num_label_chunks = 0
        self.num_correct_chunks = 0

    def name(self):
        """
        Return name of metric instance.
        """
        return "precision", "recall", "f1"


if __name__ == "__main__":
    metric = ChunkEvaluator(label_list=[])
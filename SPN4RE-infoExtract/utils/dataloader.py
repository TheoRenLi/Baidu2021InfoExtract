import torch
from torch.utils.data import Dataset

class DataCollator:
    """
        Functions: 
            used for torch.utils.data.DataLoader's parameter: collate_fn.
            return processed data through __call__ func.
        Return: 
            data returned format is free to define.
        Examples:

            from torch.utils.data import Dataset, DataLoader

            train_datas = [...]
            train_labels = [...]
            train_ds = Dataset(train_datas, train_labels)
            dataloader = DataLoader(train_ds, batch_size=5, shuffle=True, collate_fn=DataCollator(), num_workers=2)
            for i, formatted_data in enumerate(dataloader):
                ...
    """

    def __init__(self):
        pass

    def __call__(self, batch):
        return self._collate(batch)

    def _collate(self, batch):
        batch_size = len(batch)
        is_test = len(batch[0]) == 2

        sent_idx = [ele[0] for ele in batch]
        sent_ids = [ele[1] for ele in batch]
        if not is_test:
            targets = [ele[2] for ele in batch]

        sent_lens = list(map(len, sent_ids))
        max_sent_len = max(sent_lens)

        input_ids = torch.zeros((batch_size, max_sent_len), requires_grad=False).long()
        attention_mask = torch.zeros((batch_size, max_sent_len), requires_grad=False, dtype=torch.float32)
        for idx, (seq, seqlen) in enumerate(zip(sent_ids, sent_lens)):
            input_ids[idx, :seqlen] = torch.LongTensor(seq)
            attention_mask[idx, :seqlen] = torch.FloatTensor([1] * seqlen)
            if not is_test:
                targets = [{k: torch.tensor(v, dtype=torch.long, requires_grad=False) for k, v in t.items()} for t in targets]
        info = {"seq_len": sent_lens, "sent_idx": sent_idx}

        return (input_ids, attention_mask, targets, info) if not is_test else (input_ids, attention_mask, info)
        


class TextDataset(Dataset):
    """
        Re-construct Dataset class.
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
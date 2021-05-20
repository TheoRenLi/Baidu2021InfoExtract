import torch.nn as nn
import torch
from models.set_decoder import SetDecoder
from models.seq_encoder import SeqEncoder


class SetPred4RE(nn.Module):

    def __init__(self, args, num_classes):
        super(SetPred4RE, self).__init__()
        self.args = args
        self.encoder = SeqEncoder(args)
        config = self.encoder.config
        self.num_classes = num_classes
        self.decoder = SetDecoder(config, args.num_generated_triples, args.num_decoder_layers, num_classes, return_intermediate=False)

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.encoder(input_ids, attention_mask)
        # class_logits: batch x 10 x classes
        # head_start_logits: batch x 10 x batch_max_length
        class_logits, head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = self.decoder(encoder_hidden_states=last_hidden_state, encoder_attention_mask=attention_mask)
        # head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = span_logits.split(1, dim=-1)
        head_start_logits = head_start_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        head_end_logits = head_end_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        tail_start_logits = tail_start_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        tail_end_logits = tail_end_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)# [bsz, num_generated_triples, seq_len]
        outputs = {'pred_rel_logits': class_logits, 'head_start_logits': head_start_logits, 'head_end_logits': head_end_logits, 'tail_start_logits': tail_start_logits, 'tail_end_logits': tail_end_logits}
        return outputs






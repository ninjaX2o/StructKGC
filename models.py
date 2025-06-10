from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from triplet_mask import construct_mask


def build_model(args) -> nn.Module:
    return CustomBertModel(args)


@dataclass
class ModelOutput:
    logits_sp: Union[torch.tensor, None]
    logits: Union[torch.tensor, None]
    logits_hr_hp: torch.tensor 
    logits_hpt: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    mask: torch.tensor


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)

        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.tail_bert = deepcopy(self.hr_bert)


    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        cls_output = nn.functional.normalize(cls_output, dim=1)

        return cls_output

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                hrp_token_ids, hrp_mask, hrp_token_type_ids,
                tail_token_ids=None, tail_mask=None, tail_token_type_ids=None,
                head_token_ids=None, head_mask=None, head_token_type_ids=None,
                negative_token_ids=None, negative_mask=None, negative_token_type_ids=None,
                only_ent_embedding=False, only_head_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)
        if only_head_embedding:
            return self.predict_head_embedding(hr_token_ids=hr_token_ids,
                                               hr_mask=hr_mask,
                                               hr_token_type_ids=hr_token_type_ids)

        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)
        hp_vector = self._encode(self.hr_bert,
                                 token_ids=hrp_token_ids,
                                 mask=hrp_mask,
                                 token_type_ids=hrp_token_type_ids)

        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)


        if negative_token_ids is not None:
            negative_vector = self._encode(self.tail_bert,
                                           token_ids=negative_token_ids,
                                           mask=negative_mask,
                                           token_type_ids=negative_token_type_ids)

        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'hp_vector': hp_vector,
                # 'head_vector': head_vector.to('cuda:0') if head_token_ids is not None else None,
                'negative_vector': negative_vector if negative_token_ids is not None else None,}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        hp_vector= output_dict['hp_vector']
        labels = torch.arange(batch_size).to(hr_vector.device)
        #cl
        logits = hr_vector.mm(tail_vector.t())
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()
        logits_hpt = hp_vector.mm(tail_vector.t())
        if self.training:
            logits_hpt -= torch.zeros(logits_hpt.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits_hpt *= self.log_inv_t.exp()
        triplet_mask_cl = batch_dict.get('triplet_mask_cl', None)
        if triplet_mask_cl is not None:
            logits.masked_fill_(~triplet_mask_cl, -1e4)
        #sp
        logits_sp = hr_vector.mm(tail_vector.t())
        logits_hr_hp = hr_vector.mm(hp_vector.t())*self.log_inv_t.exp()
        triplet_mask = batch_dict.get('triplet_mask', None)


        if self.training:
            logits_sp[~triplet_mask] -= self.add_margin
        logits_sp *= self.log_inv_t.exp()


        if output_dict['negative_vector'] is not None:
            negative_vector = output_dict['negative_vector']
            if negative_vector.dim() == 3:
                num_negs = negative_vector.size(1)
                negative_vector = torch.reshape(negative_vector, [batch_size * num_negs, -1])
            negative_logits_sp = hr_vector.mm(negative_vector.t())
            negative_logits_sp *= self.log_inv_t.exp()
            negative_logits_cl = hr_vector.mm(negative_vector.t())
            negative_logits_cl *= self.log_inv_t.exp()
            triplet_negative_mask = batch_dict.get('triplet_negative_mask', None)
            if triplet_negative_mask is not None:
                negative_logits_sp[~triplet_negative_mask]-= self.add_margin
                negative_logits_cl.masked_fill_(~triplet_negative_mask, -1e4)
            logits_sp = torch.cat([logits_sp, negative_logits_sp], dim=-1)
            logits = torch.cat([logits, negative_logits_cl], dim=-1)
            triplet_mask=torch.cat([triplet_mask, triplet_negative_mask], dim=-1)

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

        return {'logits_sp': logits_sp,
                'logits': logits,
                'logits_hpt': logits_hpt,
                'labels': labels,
                'logits_hr_hp': logits_hr_hp,
                'mask': ~triplet_mask if triplet_mask is not None else None,
                'inv_t': self.log_inv_t.detach().exp()}

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}

    @torch.no_grad()
    def predict_head_embedding(self, hr_token_ids, hr_mask, hr_token_type_ids, **kwargs) -> dict:
        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)
        return {'hr_vector': hr_vector}


def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    return output_vector

import json
import os
from time import time
from typing import List, Tuple

import torch
import tqdm
from dataclasses import dataclass, asdict

from config import args
from dict_hub import get_entity_dict, get_all_triplet_dict
from doc_w import load_data, Example
from logger_config import logger
from predict import BertPredictor
from rerank import rerank_by_graph
from triplet import EntityDict


def _setup_entity_dict() -> EntityDict:
    if args.task == 'wiki5m_ind':
        return EntityDict(entity_dict_dir=os.path.dirname(args.valid_path),
                          inductive_test_path=args.valid_path)
    return get_entity_dict()


entity_dict = _setup_entity_dict()
all_triplet_dict = get_all_triplet_dict()

@dataclass
class PredInfo:
    head: str
    relation: str
    tail: str
    pred_tail: str
    pred_score: float
    topk_score_info: str
    gold_score_info: str
    rank: int
    correct: bool


@torch.no_grad()
def compute_metrics(hr_tensor: torch.tensor,hp_tensor: torch.tensor,
                    entities_tensor: torch.tensor,
                    target: List[int],
                    examples: List[Example],
                    k=3, batch_size=256) -> Tuple:
    assert hr_tensor.size(1) == entities_tensor.size(1)
    total = hr_tensor.size(0)
    entity_cnt = len(entity_dict)
    assert entity_cnt == entities_tensor.size(0)
    target = torch.LongTensor(target).unsqueeze(-1).to(hr_tensor.device)
    topk_scores, topk_indices = [], []
    ranks = []

    mean_rank, mrr, hit1, hit3, hit10, hit50 = 0, 0, 0, 0, 0, 0
    golds = []
    gold_scores = []
    for start in tqdm.tqdm(range(0, total, batch_size)):
        end = start + batch_size
        # batch_size * entity_cnt
        batch_score = torch.mm(hr_tensor[start:end, :], entities_tensor.t())*0.4
        batch_score += torch.mm(hp_tensor[start:end, :], entities_tensor.t())*1.2
        assert entity_cnt == batch_score.size(1)
        batch_target = target[start:end]

        # re-ranking based on topological structure
        if args.task.lower() != 'dbpedia500':
            rerank_by_graph(batch_score, examples[start:end], entity_dict=entity_dict)
        # filter known triplets
        for idx in range(batch_score.size(0)):
            mask_indices = []
            cur_ex = examples[start + idx]
            gold_neighbor_ids = all_triplet_dict.get_neighbors(cur_ex.head_id, cur_ex.relation)
            if len(gold_neighbor_ids) > 10000:
                logger.debug('{} - {} has {} neighbors'.format(cur_ex.head_id, cur_ex.relation, len(gold_neighbor_ids)))
            for e_id in gold_neighbor_ids:
                if e_id == cur_ex.tail_id:
                    continue
                mask_indices.append(entity_dict.entity_to_idx(e_id))
            mask_indices = torch.LongTensor(mask_indices).to(batch_score.device)
            golds.append(mask_indices.tolist())
            gold_scores.append(batch_score[idx][mask_indices].tolist())
            batch_score[idx].index_fill_(0, mask_indices, -1)

        batch_sorted_score, batch_sorted_indices = torch.sort(batch_score, dim=-1, descending=True)
        target_rank = torch.nonzero(batch_sorted_indices.eq(batch_target).long(), as_tuple=False)
        assert target_rank.size(0) == batch_score.size(0)
        for idx in range(batch_score.size(0)):
            idx_rank = target_rank[idx].tolist()
            assert idx_rank[0] == idx
            cur_rank = idx_rank[1]

            # 0-based -> 1-based
            cur_rank += 1
            mean_rank += cur_rank
            mrr += 1.0 / cur_rank
            hit1 += 1 if cur_rank <= 1 else 0
            hit3 += 1 if cur_rank <= 3 else 0
            hit10 += 1 if cur_rank <= 10 else 0
            hit50 += 1 if cur_rank <= 50 else 0
            ranks.append(cur_rank)

        topk_scores.extend(batch_sorted_score[:, :k].tolist())
        topk_indices.extend(batch_sorted_indices[:, :k].tolist())

    metrics = {'mean_rank': mean_rank, 'mrr': mrr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10, 'hit@50': hit50}
    metrics = {k: round(v / total, 4) for k, v in metrics.items()}
    assert len(topk_scores) == total
    return topk_scores, topk_indices, metrics, ranks,golds,gold_scores


def predict_by_split():
    assert os.path.exists(args.valid_path)
    assert os.path.exists(args.train_path)

    predictor = BertPredictor()
    predictor.load(ckt_path=args.eval_model_path, use_data_parallel=True)
    entity_tensor = predictor.predict_by_entities(entity_dict.entity_exs)

    entity_tensor_path = '{}/entity_tensor'.format(args.model_dir)
    torch.save(entity_tensor, entity_tensor_path)

    forward_metrics = eval_single_direction(predictor,
                                            entity_tensor=entity_tensor,
                                            eval_forward=True,
                                            batch_size=512)
    backward_metrics = eval_single_direction(predictor,
                                             entity_tensor=entity_tensor,
                                             eval_forward=False,
                                             batch_size=512)
    metrics = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}
    logger.info('Averaged metrics: {}'.format(metrics))

    prefix, basename = os.path.dirname(args.eval_model_path), os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)
    with open('{}/metrics_{}_{}.json'.format(prefix, split, basename), 'w', encoding='utf-8') as writer:
        writer.write('forward metrics: {}\n'.format(json.dumps(forward_metrics)))
        writer.write('backward metrics: {}\n'.format(json.dumps(backward_metrics)))
        writer.write('average metrics: {}\n'.format(json.dumps(metrics)))


def eval_single_direction(predictor: BertPredictor,
                          entity_tensor: torch.tensor,
                          eval_forward=True,
                          batch_size=256) -> dict:
    start_time = time()
    examples = load_data(args.valid_path, add_forward_triplet=eval_forward, add_backward_triplet=not eval_forward)

    # hr_tensor = predictor.predict_by_examples(examples, only_head_embedding=True)
    hr_tensor,t_tensor ,hp_tensor= predictor.predict_by_examples(examples, only_head_embedding=False)

    def lalign(x, y, alpha=2):
        return (x - y).norm(dim=1).pow(alpha).mean()


    def lunif(x, t=2):
        sq_pdist = torch.pdist(x, p=2).pow(2)
        return sq_pdist.mul(-t).exp().mean().log()

    align = lalign(
        hr_tensor.to("cpu"),
        t_tensor.to("cpu")
        ).item()
    unif = lunif(hr_tensor.to("cpu") ).item()+lunif(t_tensor.to("cpu") ).item()
    print("hr_tensor.shape {}".format(hr_tensor.shape))
    print("t_tensor.shape {}".format(t_tensor.shape))
    print("align {}".format(align))
    print("unif {}".format(unif))
    eval_dir = 'forward' if eval_forward else 'backward'
    hr_tensor_path = '{}/{}_hr_tensor'.format(args.model_dir, eval_dir)
    torch.save(hr_tensor, hr_tensor_path)

    hr_tensor = hr_tensor.to(entity_tensor.device)
    hp_tensor = hp_tensor.to(entity_tensor.device)

    target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]
    logger.info('predict tensor done, compute metrics...')

    topk_scores, topk_indices, metrics, ranks,golds,gold_scores = compute_metrics(hr_tensor=hr_tensor, hp_tensor=hp_tensor,entities_tensor=entity_tensor,
                                                                target=target, examples=examples,
                                                                batch_size=batch_size,k=20)
    eval_dir = 'forward' if eval_forward else 'backward'
    logger.info('{} metrics: {}'.format(eval_dir, json.dumps(metrics)))

    pred_infos = []
    for idx, ex in enumerate(examples):
        cur_topk_scores = topk_scores[idx]
        cur_topk_indices = topk_indices[idx]
        cur_gold_scores = gold_scores[idx]
        cur_gold_indices = golds[idx]
        pred_idx = cur_topk_indices[0]
        cur_score_info = {entity_dict.get_entity_by_idx(topk_idx).entity: round(topk_score, 3)
                          for topk_score, topk_idx in zip(cur_topk_scores, cur_topk_indices)}
        gold_score_info = {entity_dict.get_entity_by_idx(topk_idx).entity: round(topk_score, 3)
                          for topk_score, topk_idx in zip(cur_gold_scores, cur_gold_indices)}
        pred_info = PredInfo(head=ex.head, relation=ex.relation,
                             tail=ex.tail, pred_tail=entity_dict.get_entity_by_idx(pred_idx).entity,
                             pred_score=round(cur_topk_scores[0], 4),
                             topk_score_info=json.dumps(cur_score_info),
                             gold_score_info=json.dumps(gold_score_info),
                             rank=ranks[idx],
                             correct=pred_idx == target[idx])
        pred_infos.append(pred_info)

    prefix, basename = os.path.dirname(args.eval_model_path), os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)
    with open('{}/eval_{}_{}_{}.json'.format(prefix, split, eval_dir, basename), 'w', encoding='utf-8') as writer:
        writer.write(json.dumps([asdict(info) for info in pred_infos], ensure_ascii=False, indent=4))

    logger.info('Evaluation takes {} seconds'.format(round(time() - start_time, 3)))
    return metrics


if __name__ == '__main__':
    predict_by_split()

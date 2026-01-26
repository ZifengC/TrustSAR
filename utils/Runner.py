import gc
import logging
import time
from typing import Dict, List
from tqdm import tqdm

import numpy as np
import torch
import torch.optim
from torch.utils.data import DataLoader

from models import BaseModel
from utils import const, utils

from .dataset import *


class BaseRunner(object):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch',
                            type=int,
                            default=1,
                            help='Number of epochs.')
        parser.add_argument('--lr',
                            type=float,
                            default=1e-3,
                            help='Learning rate.')
        parser.add_argument(
            '--patience',
            type=int,
            default=3,
            help=
            'Number of epochs with no improvement after which learning rate will be reduced'
        )
        parser.add_argument(
            '--early_stop',
            type=int,
            default=5,
            help='The number of epochs when dev results drop continuously.')
        parser.add_argument('--min_lr', type=float, default=1e-6)
        parser.add_argument('--l2',
                            type=float,
                            default=3e-5,
                            help='Weight decay in optimizer.')

        parser.add_argument('--infoNCE_neg_sample', type=int, default=1024)

        parser.add_argument('--batch_size',
                            type=int,
                            default=1024,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size',
                            type=int,
                            default=512,
                            help='Batch size during testing.')
        parser.add_argument(
            '--num_workers',
            type=int,
            default=10,
            help='Number of processors when prepare batches in DataLoader')

        return parser

    def __init__(self, args) -> None:

        self.epoch = args.epoch
        self.print_interval = 500

        self.early_stop = args.early_stop
        self.learning_rate = args.lr
        self.patience = args.patience
        self.min_lr = args.min_lr
        self.l2 = args.l2
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.num_workers = args.num_workers

        self.infoNCE_neg_sample = args.infoNCE_neg_sample

        self.topk = [1, 5, 10, 20, 50]
        self.metrics = ['NDCG', 'HR']
        # early stop based on main_metric
        self.main_metric = 'NDCG@5'

        self.train_loader: DataLoader = None
        self.val_loader: DataLoader = None
        self.test_loader: DataLoader = None
        self.optimizer: torch.optim.Optimizer = None

        self.datasetParaDict = {
            "user_vocab": None,
        }
        self.query_vocab = None

    def _build_optimizer(self, model: BaseModel):
        self.optimizer = torch.optim.Adam(model.customize_parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=self.l2)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=self.patience,
            min_lr=self.min_lr,
            verbose=True)

    def getDataLoader(self, dataset: BaseDataSet, batch_size: int,
                      shuffle: bool) -> DataLoader:
        dataloader = DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=batch_size // self.num_workers + 1,
            worker_init_fn=utils.worker_init_fn,
            persistent_workers=True,
            collate_fn=dataset.collate_batch)
        return dataloader

    def set_dataloader(self):
        raise NotImplementedError

    def train(self, model: BaseModel):
        self._build_optimizer(model)

        if model.query_item_alignment:
            self.get_query_vocab()
            self.InfoNCE_dataloader = self.getDataLoader(
                InfoNCEDataset(query_vocab=self.query_vocab),
                self.infoNCE_neg_sample,
                shuffle=False)

        main_metric_results, dev_results = list(), list()
        for epoch in range(self.epoch):
            gc.collect()
            torch.cuda.empty_cache()

            epoch_loss = self.train_epoch(epoch, model)

            logging.info("epoch:{} mean loss:{:.4f}".format(epoch, epoch_loss))

            dev_result, main_result = self.evaluate(model, 'val')
            dev_results.append(dev_result)
            main_metric_results.append(main_result)
            logging.info("Dev Result:")
            logging.info(dev_result)
            print("Dev Result: {}".format(dev_result))
            self.scheduler.step(main_result)

            if max(main_metric_results) == main_metric_results[-1]:
                model.save_model()
                test_result, _ = self.evaluate(model, 'test')
                logging.info("Test Result:")
                logging.info(test_result)
                print("Test Result: {}".format(test_result))

            if self.early_stop > 0 and self.eval_termination(
                    main_metric_results):
                logging.info('Early stop at %d based on dev result.' %
                             (epoch + 1))
                break

        best_epoch = main_metric_results.index(max(main_metric_results))
        logging.info(" ")
        logging.info("Best Dev Result at epoch:{}".format(best_epoch))
        logging.info(dev_results[best_epoch])
        print("\nBest Dev Result at epoch: {}\n {}".format(
            best_epoch, dev_results[best_epoch]))

        model.load_model()

        test_result, _ = self.evaluate(model, 'test')
        logging.info(" ")
        logging.info("Test Result:")
        logging.info(test_result)
        print("\nTest Result: {}".format(test_result))

    def train_epoch(self, epoch: int, model: BaseModel):
        raise NotImplementedError

    def eval_termination(self, criterion: List[float]) -> bool:
        if len(criterion) - criterion.index(max(criterion)) > self.early_stop:
            return True
        return False

    @staticmethod
    def evaluate_method(predictions: np.ndarray, topk: int,
                        metrics: List[str]) -> Dict[str, float]:
        evaluations = dict()
        sort_idx = (-predictions).argsort(axis=1)
        gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1
        for k in topk:
            hit = (gt_rank <= k)
            for metric in metrics:
                key = '{}@{}'.format(metric, k)
                if metric == 'HR':
                    evaluations[key] = hit.mean()
                elif metric == 'NDCG':
                    evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
                else:
                    raise ValueError(
                        'Undefined evaluation metric: {}.'.format(metric))
        return evaluations

    @staticmethod
    @torch.no_grad()
    def predict(model: BaseModel,
                test_loader: DataLoader,
                return_sim: bool = False,
                return_hits: bool = False,
                max_hit_log: int = 20):
        model.eval()
        predictions = list()
        sim_list = list() if return_sim else None
        hit_info = [] if return_hits else None

        start = time.time()
        for step, batch in enumerate(test_loader):
            batch_gpu = utils.batch_to_gpu(batch, model.device)
            prediction = model.predict(batch_gpu)
            predictions.extend(prediction.cpu().data.numpy())
            if return_sim:
                sims = BaseRunner.compute_query_last_sim(model, batch_gpu)
                sim_list.extend(sims.cpu().numpy().tolist())
            if return_hits and batch_gpu.get('search', False):
                scores = prediction
                if not torch.is_tensor(scores):
                    scores = torch.tensor(scores, device=model.device)
                gt = scores[:, 0:1]
                rank = (scores > gt).sum(dim=1) + 1
                hit_mask = rank == 1
                if hit_mask.any() and len(hit_info) < max_hit_log:
                    users = batch_gpu['user'].squeeze(-1)[hit_mask]
                    items = batch_gpu['item'].squeeze(-1)[hit_mask]
                    queries = batch_gpu['query'][hit_mask]
                    for u, it, q in zip(users.tolist(),
                                        items.tolist(),
                                        queries.cpu().numpy().tolist()):
                        hit_info.append({
                            "user": u,
                            "item": it,
                            "query": q
                        })
                        if len(hit_info) >= max_hit_log:
                            break

        logging.info("model evaluate time used:{}s".format(time.time() -
                                                           start))
        predictions = np.array(predictions)

        sims_arr = np.array(sim_list) if return_sim else None
        if return_sim or return_hits:
            return predictions, sims_arr, hit_info
        return predictions

    @staticmethod
    @torch.no_grad()
    def compute_query_last_sim(model: BaseModel, batch: dict) -> torch.Tensor:
        """
        仅用于搜索数据：计算当前 query 与最近一次交互（search 用 query，rec 用 item）的余弦相似度。
        """
        if 'query' not in batch or 'all_his' not in batch or 'all_his_type' not in batch:
            return torch.zeros((batch['batch_size'], ), device=model.device)

        query_emb = model.session_embedding.get_query_emb(batch['query'])
        all_his = batch['all_his']
        all_his_type = batch['all_his_type']
        all_his_ts = batch.get('all_his_ts', None)

        if all_his_ts is not None:
            valid_mask = torch.isfinite(all_his_ts)
        else:
            valid_mask = all_his != 0

        valid_lens = valid_mask.sum(dim=1)
        last_idx = (valid_lens - 1).clamp(min=0)
        batch_indices = torch.arange(all_his.size(0), device=all_his.device)
        last_item = all_his[batch_indices, last_idx]
        last_type = all_his_type[batch_indices, last_idx]

        last_emb = torch.zeros_like(query_emb)
        rec_mask = last_type == 1
        if rec_mask.any():
            rec_ids = last_item[rec_mask]
            rec_emb = model.session_embedding.get_item_emb(rec_ids)
            last_emb[rec_mask] = rec_emb

        src_mask = last_type == 2
        if src_mask.any():
            src_ids = last_item[src_mask].long()
            keyword_map = model.session_embedding.map_vocab['keyword'].to(
                model.device)
            src_query_tokens = keyword_map[src_ids]
            src_query_emb = model.session_embedding.get_query_emb(
                src_query_tokens)
            last_emb[src_mask] = src_query_emb

        sims = torch.nn.functional.cosine_similarity(query_emb,
                                                     last_emb,
                                                     dim=-1)
        return sims

    def evaluate(self, model: BaseModel, mode: str):
        raise NotImplementedError

    def build_dataset(self):
        if self.datasetParaDict['user_vocab'] is None:
            self.datasetParaDict['user_vocab'] = utils.load_pickle(
                const.user_vocab)

    def get_query_vocab(self):
        if self.query_vocab is None:
            self.query_vocab = utils.load_pickle(const.query_vocab)


class SarRunner(BaseRunner):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--src_loss_weight', type=float, default=0.1)

        return BaseRunner.parse_runner_args(parser)

    def __init__(self, args) -> None:
        super().__init__(args)
        self.build_dataset()
        self.set_dataloader()

        self.src_loss_weight = args.src_loss_weight

    def set_dataloader(self):
        self.rec_train_loader = self.getDataLoader(self.traindata['rec'],
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        self.rec_val_loader = self.getDataLoader(
            self.valdata['rec'],
            batch_size=self.eval_batch_size,
            shuffle=False)
        self.rec_test_loader = self.getDataLoader(
            self.testdata['rec'],
            batch_size=self.eval_batch_size,
            shuffle=False)

        src_train_batch_size = len(self.traindata['src']) // (
            len(self.traindata['rec']) // self.batch_size + 1) + 1
        logging.info('src train batch size:{}'.format(src_train_batch_size))
        self.src_train_loader = self.getDataLoader(
            self.traindata['src'],
            batch_size=src_train_batch_size,
            shuffle=True)
        self.src_val_loader = self.getDataLoader(
            self.valdata['src'],
            batch_size=self.eval_batch_size,
            shuffle=False)
        self.src_test_loader = self.getDataLoader(
            self.testdata['src'],
            batch_size=self.eval_batch_size,
            shuffle=False)

    def build_dataset(self):
        super().build_dataset()
        self.traindata = {
            "rec":
            RecDataSet(train='train',
                       user_vocab=self.datasetParaDict['user_vocab']),
            "src":
            SrcDataSet(train='train',
                       user_vocab=self.datasetParaDict['user_vocab'])
        }
        self.valdata = {
            "rec":
            RecDataSet(train='val',
                       user_vocab=self.datasetParaDict['user_vocab']),
            "src":
            SrcDataSet(train='val',
                       user_vocab=self.datasetParaDict['user_vocab'])
        }
        self.testdata = {
            "rec":
            RecDataSet(train='test',
                       user_vocab=self.datasetParaDict['user_vocab']),
            "src":
            SrcDataSet(train='test',
                       user_vocab=self.datasetParaDict['user_vocab'])
        }

    def train_epoch(self, epoch: int, model: BaseModel):
        model.train()
        logging.info(" ")
        logging.info("Epoch: {}".format(epoch))
        print("\nEpoch: {}".format(epoch))

        if model.query_item_alignment:
            InfoNCE_iterator = iter(self.InfoNCE_dataloader)

        src_iterator = iter(self.src_train_loader)

        loss_list = []
        loss_dict = {"rec": {}, "src": {}}
        start = time.time()
        for step, rec_batch in enumerate(tqdm(self.rec_train_loader)):
            try:
                src_batch = next(src_iterator)
            except StopIteration:
                src_iterator = iter(self.src_train_loader)
                src_batch = next(src_iterator)

            if model.query_item_alignment:
                align_dict = next(InfoNCE_iterator)
                rec_batch.update(align_dict)
                src_batch.update(align_dict)

            rec_loss = model.loss(utils.batch_to_gpu(rec_batch, model.device))
            src_loss = model.src_loss(
                utils.batch_to_gpu(src_batch, model.device))

            for k in rec_loss.keys():
                if k in loss_dict['rec'].keys():
                    loss_dict['rec'][k].append(rec_loss[k].item())
                    loss_dict['src'][k].append(src_loss[k].item())
                else:
                    loss_dict['rec'][k] = [rec_loss[k].item()]
                    loss_dict['src'][k] = [src_loss[k].item()]

            total_loss = rec_loss['total_loss'] + \
                src_loss['total_loss'] * self.src_loss_weight

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            loss_list.append(total_loss.item())

            if step > 0 and step % self.print_interval == 0:
                logging.info(
                    "epoch:{:d} step:{:d} time:{:.2f}s |rec {}| |src {}|".
                    format(
                        epoch, step,
                        time.time() - start, " ".join([
                            "{}:{:.4f}".format(k,
                                               np.mean(v).item())
                            for k, v in loss_dict['rec'].items()
                        ]), " ".join([
                            "{}:{:.4f}".format(k,
                                               np.mean(v).item())
                            for k, v in loss_dict['src'].items()
                        ])))
        logging.info("total time: {:.2f}s".format(time.time() - start))

        return np.mean(loss_list).item()

    def evaluate(self, model: BaseModel, mode: str):
        if mode == 'val':
            rec_predictions = self.predict(model, self.rec_val_loader)
            src_predictions, src_sims, _ = self.predict(model,
                                                        self.src_val_loader,
                                                        return_sim=True,
                                                        return_hits=False)
        elif mode == 'test':
            rec_predictions = self.predict(model, self.rec_test_loader)
            src_predictions, src_sims, hit_info = self.predict(
                model,
                self.src_test_loader,
                return_sim=True,
                return_hits=True,
                max_hit_log=20)
        else:
            raise ValueError('test set error')
        rec_results = self.evaluate_method(rec_predictions, self.topk,
                                           self.metrics)
        src_results = self.evaluate_method(src_predictions, self.topk,
                                           self.metrics)

        results = {
            "rec": utils.format_metric(rec_results),
            "src": utils.format_metric(src_results)
        }
        if mode == 'test' and 'hit_info' in locals() and hit_info:
            logging.info("Sample search hits (top1 correct) [max 20]:")
            for h in hit_info:
                logging.info(f"user={h['user']} item={h['item']} query={h['query']}")

        if src_sims is not None and len(src_sims) > 0:
            quantiles = np.quantile(src_sims, [1 / 3, 2 / 3])
            buckets = [
                src_sims <= quantiles[0],
                (src_sims > quantiles[0]) & (src_sims <= quantiles[1]),
                src_sims > quantiles[1]
            ]
            bucket_keys = ['low', 'mid', 'high']
            for mask, bkey in zip(buckets, bucket_keys):
                if mask.sum() == 0:
                    continue
                bucket_eval = self.evaluate_method(
                    src_predictions[mask], self.topk, self.metrics)
                results[f"src_{bkey}"] = utils.format_metric(bucket_eval)

        return results, (rec_results[self.main_metric] +
                         src_results[self.main_metric]) / 2.0

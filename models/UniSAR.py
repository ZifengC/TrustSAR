import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List

from utils import const

from .BaseModel import BaseModel
from .layers import FullyConnectedLayer, feature_align, PositionalEmbedding, PLE_layer


class UniSAR(BaseModel):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--num_heads', type=int, default=2)

        parser.add_argument('--q_i_cl_temp', type=float, default=0.5)
        parser.add_argument('--q_i_cl_weight', type=float, default=0.001)

        parser.add_argument('--his_cl_temp', type=float, default=0.1)
        parser.add_argument('--his_cl_weight', type=float, default=0.1)

        parser.add_argument('--pred_hid_units',
                            type=List,
                            default=[200, 80, 1])

        parser.add_argument('--memory_eps', type=float, default=0.001)
        parser.add_argument('--memory_bias_min', type=float, default=0.8)
        parser.add_argument('--memory_bias_max', type=float, default=1.2)
        parser.add_argument('--memory_log', action='store_true')
        parser.add_argument('--memory_log_interval', type=int, default=200)
        parser.add_argument('--gate_floor', type=float, default=0.85,
                            help='Lower bound for gate blending to avoid hard suppression')

        return BaseModel.parse_model_args(parser)

    def __init__(self, args):
        super().__init__(args)
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.batch_size = args.batch_size

        self.src_pos = PositionalEmbedding(const.max_src_session_his_len,
                                           self.item_size)
        self.rec_pos = PositionalEmbedding(const.max_rec_his_len,
                                           self.item_size)
        self.global_pos_emb = PositionalEmbedding(
            const.max_rec_his_len + const.max_src_session_his_len,
            self.item_size)

        self.rec_transformer = Transformer(emb_size=self.item_size,
                                           num_heads=self.num_heads,
                                           num_layers=self.num_layers,
                                           dropout=self.dropout)
        self.src_transformer = Transformer(emb_size=self.item_size,
                                           num_heads=self.num_heads,
                                           num_layers=self.num_layers,
                                           dropout=self.dropout)
        self.global_transformer = Transformer(emb_size=self.item_size,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_layers,
                                              dropout=self.dropout)

        self.q_i_cl_temp = args.q_i_cl_temp
        self.q_i_cl_weight = args.q_i_cl_weight
        if self.q_i_cl_weight > 0:
            self.query_item_alignment = True
            self.feature_alignment = feature_align(self.q_i_cl_temp,
                                                   self.item_size)

        self.his_cl_temp = args.his_cl_temp
        self.his_cl_weight = args.his_cl_weight
        if self.his_cl_weight > 0:
            self.rec_his_cl = TransAlign(batch_size=self.batch_size,
                                         hidden_dim=self.item_size,
                                         device=self.device,
                                         infoNCE_temp=self.his_cl_temp)
            self.src_his_cl = TransAlign(batch_size=self.batch_size,
                                         hidden_dim=self.item_size,
                                         device=self.device,
                                         infoNCE_temp=self.his_cl_temp)

        self.rec_decoder_layer = MemoryTransformerDecoderLayer(
            d_model=self.item_size,
            nhead=self.num_heads,
            dim_feedforward=self.item_size,
            dropout=self.dropout,
            batch_first=True)
        self.rec_cross_fusion = MemoryTransformerDecoder(
            self.rec_decoder_layer, num_layers=self.num_layers)

        self.src_decoder_layer = MemoryTransformerDecoderLayer(
            d_model=self.item_size,
            nhead=self.num_heads,
            dim_feedforward=self.item_size,
            dropout=self.dropout,
            batch_first=True)
        self.src_cross_fusion = MemoryTransformerDecoder(
            self.src_decoder_layer, num_layers=self.num_layers)

        self.rec_his_attn_pooling = Target_Attention(self.item_size,
                                                     self.item_size)
        self.src_his_attn_pooling = Target_Attention(self.item_size,
                                                     self.item_size)

        self.rec_query = torch.nn.parameter.Parameter(torch.randn(
            (1, self.query_size), requires_grad=True),
                                                      requires_grad=True)
        nn.init.xavier_normal_(self.rec_query)

        self.hidden_unit = args.pred_hid_units

        input_dim = 3 * self.item_size + self.user_size + self.query_size
        self.ple_layer = PLE_layer(orig_input_dim=input_dim,
                                   bottom_mlp_dims=[64],
                                   tower_mlp_dims=[128, 64],
                                   task_num=2,
                                   shared_expert_num=4,
                                   specific_expert_num=4,
                                   dropout=self.dropout)
        self.rec_fc_layer = FullyConnectedLayer(input_size=64,
                                                hidden_unit=self.hidden_unit,
                                                batch_norm=False,
                                                sigmoid=True,
                                                activation='relu',
                                                dropout=self.dropout)
        self.src_fc_layer = FullyConnectedLayer(input_size=64,
                                                hidden_unit=self.hidden_unit,
                                                batch_norm=False,
                                                sigmoid=True,
                                                activation='relu',
                                                dropout=self.dropout)

        self.loss_fn = nn.BCELoss()
        self.memory_eps = args.memory_eps
        # 双分支信任记忆：src 用于 memory=rec2src / tgt=src2src，rec 用于 memory=src2rec / tgt=rec2rec
        self.src_memory_trust_memory = SlowTrustMemory(dim=self.item_size,
                                                       epsilon=self.memory_eps,
                                                       clamp_min=args.memory_bias_min,
                                                       clamp_max=args.memory_bias_max)
        self.src_tgt_trust_memory = SlowTrustMemory(dim=self.item_size,
                                                    epsilon=self.memory_eps,
                                                    clamp_min=args.memory_bias_min,
                                                    clamp_max=args.memory_bias_max)
        self.rec_memory_trust_memory = SlowTrustMemory(dim=self.item_size,
                                                       epsilon=self.memory_eps,
                                                       clamp_min=args.memory_bias_min,
                                                       clamp_max=args.memory_bias_max)
        self.rec_tgt_trust_memory = SlowTrustMemory(dim=self.item_size,
                                                    epsilon=self.memory_eps,
                                                    clamp_min=args.memory_bias_min,
                                                    clamp_max=args.memory_bias_max)
        self.memory_log = args.memory_log
        self.memory_log_interval = args.memory_log_interval
        self._memory_log_counter = 0
        self._last_trust_bias = None
        self._init_weights()
        self.to(self.device)

    def src_feat_process(self, src_feat):
        query_emb, q_click_item_emb, click_item_mask = src_feat

        q_i_align_used = [query_emb, click_item_mask, q_click_item_emb]

        mean_click_item_emb = torch.sum(torch.mul(
            q_click_item_emb, click_item_mask.unsqueeze(-1)),
                                        dim=-2)  # batch, max_src_len, dim
        mean_click_item_emb = mean_click_item_emb / (torch.max(
            click_item_mask.sum(-1, keepdim=True),
            torch.ones_like(click_item_mask.sum(-1, keepdim=True))))
        query_his_emb = query_emb
        click_item_his_emb = mean_click_item_emb

        return query_his_emb + click_item_his_emb, q_i_align_used

    '''转成embedding的地方-> 得到all_his_emb'''
    def get_all_his_emb(self, all_his, all_his_type): 
        #对推荐行为：只取 item embedding（e_i）
        rec_his = torch.masked_fill(all_his, all_his_type != 1, 0)
        self._debug_tensor_range("rec_his_raw", rec_his)
        rec_his_emb = self.session_embedding.get_item_emb(rec_his)
        rec_his_emb = torch.masked_fill(rec_his_emb,
                                        (all_his_type != 1).unsqueeze(-1), 0)
        #对搜索行为：调用 session_embedding.forward()
        src_session_his = torch.masked_fill(all_his, all_his_type != 2, 0)
        src_limit = self.session_embedding.map_vocab['keyword'].shape[0]
        self._debug_tensor_range("src_session_his_raw",
                                 src_session_his,
                                 limit=src_limit)
        src_his_emb, q_i_align_used = self.src_feat_process(
            self.session_embedding(src_session_his))
        src_his_emb = torch.masked_fill(src_his_emb,
                                        (all_his_type != 2).unsqueeze(-1), 0)

        all_his_emb = rec_his_emb + src_his_emb
        all_his_mask = torch.where(all_his == 0, 1, 0).bool()

        return all_his_emb, all_his_mask, q_i_align_used

    def repeat_feat(self, feature_list, items_emb):
        repeat_feature_list = [
            torch.repeat_interleave(feat, items_emb.size(1), dim=0)
            for feat in feature_list
        ]
        items_emb = items_emb.reshape(-1, items_emb.size(-1))

        return repeat_feature_list, items_emb

    def _debug_tensor_range(self, tag, tensor, limit=None):
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return
        flat = tensor.reshape(-1)
        mi, ma = flat.min().item(), flat.max().item()
        if limit is None:
            limit = const.item_id_num
        if mi < 0 or ma >= limit:
            print(f"[UniSAR] {tag} 越界 shape={tensor.shape} "
                  f"dtype={tensor.dtype} min={mi} max={ma} limit={limit}")

    def _check_finite(self, tag, tensor):
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return
        finite_mask = torch.isfinite(tensor)
        if torch.all(finite_mask):
            return
        finite_vals = tensor[finite_mask]
        finite_min = finite_vals.min().item() if finite_vals.numel() > 0 else "NA"
        finite_max = finite_vals.max().item() if finite_vals.numel() > 0 else "NA"
        print(f"[UniSAR] {tag} 含非有限值 shape={tensor.shape} dtype={tensor.dtype} "
              f"nan={torch.isnan(tensor).any().item()} "
              f"inf={torch.isinf(tensor).any().item()} "
              f"finite_min={finite_min} finite_max={finite_max}")

    def match_mask_to_tensor(self, mask, tensor):
        target_len = tensor.size(1)
        if mask.size(1) == target_len:
            return mask
        if mask.size(1) > target_len:
            return mask[:, :target_len]
        pad = torch.ones(mask.size(0),
                         target_len - mask.size(1),
                         dtype=mask.dtype,
                         device=mask.device)
        return torch.cat([mask, pad], dim=1)

    def mean_pooling(self, output, his_len):
        return torch.sum(output, dim=1) / his_len.unsqueeze(-1)

    def split_rec_src(self, all_his_emb, all_his_type):
        rec_his_emb = torch.masked_select(
            all_his_emb, (all_his_type == 1).unsqueeze(-1)).reshape(
                (all_his_emb.shape[0], const.max_rec_his_len,
                 all_his_emb.shape[2]))
        src_his_emb = torch.masked_select(
            all_his_emb, (all_his_type == 2).unsqueeze(-1)).reshape(
                (all_his_emb.shape[0], const.max_src_session_his_len,
                 all_his_emb.shape[2]))
        return rec_his_emb, src_his_emb

    def forward(self, user, all_his, all_his_type, items_emb, domain):
        user_emb = self.session_embedding.get_user_emb(user)
        self._check_finite("user_emb_raw", user_emb)

        all_his_emb, all_his_mask, q_i_align_used = self.get_all_his_emb(
            all_his, all_his_type)
        self._check_finite("all_his_emb", all_his_emb)

        all_his_emb_w_pos = all_his_emb + self.global_pos_emb(all_his_emb) # Sequence Eu + positional embedding

        global_mask = all_his_type[:, :, None] == all_his_type[:, None, :]
        # 调试：查看有效历史长度
        valid_lens = (~all_his_mask).sum(dim=1)
        zero_seq = int((valid_lens == 0).sum().item())
        if zero_seq > 0:
            max_len = int(valid_lens.max().item())
            min_len = int(valid_lens.min().item())
            print(f"[UniSAR] 有 {zero_seq} 条样本无历史（min_len={min_len}, max_len={max_len})")

        global_encoded = self.global_transformer(all_his_emb_w_pos,
                                                 all_his_mask, global_mask)
        src2rec, rec2src = self.split_rec_src(global_encoded, all_his_type)
        self._check_finite("src2rec", src2rec)
        self._check_finite("rec2src", rec2src)

        rec_his_emb, src_his_emb = self.split_rec_src(all_his_emb,
                                                      all_his_type)
        rec_pad_mask = (rec_his_emb.abs().sum(dim=-1) == 0)
        src_pad_mask = (src_his_emb.abs().sum(dim=-1) == 0)
        rec_his_emb_w_pos = rec_his_emb + self.rec_pos(rec_his_emb)
        src_his_emb_w_pos = src_his_emb + self.src_pos(src_his_emb)

        rec2rec = self.rec_transformer(rec_his_emb_w_pos, rec_pad_mask)
        src2src = self.src_transformer(src_his_emb_w_pos, src_pad_mask)
        rec_pad_mask = self.match_mask_to_tensor(rec_pad_mask, rec2rec)
        src_pad_mask = self.match_mask_to_tensor(src_pad_mask, src2src)
        self._check_finite("rec2rec", rec2rec)
        self._check_finite("src2src", src2src)

        # rec 分支 gate：由 rec2rec / src2rec 与推荐点击序列相似度生成
        rec_has_click = ~rec_pad_mask
        rec_sim_numer = (rec2rec * rec_his_emb).sum(dim=-1)
        rec_sim_denom = rec2rec.norm(dim=-1) * rec_his_emb.norm(dim=-1) + 1e-8
        rec_sim = rec_sim_numer / rec_sim_denom
        rec_sim = rec_sim.masked_fill(~rec_has_click, 0.0)

        rec_tgt_write = rec2rec * rec_sim.unsqueeze(-1)
        rec_tgt_write = torch.where(
            rec_pad_mask.unsqueeze(-1),
            torch.zeros_like(rec_tgt_write),
            rec_tgt_write)
        rec_tgt_trust_bias, _ = self.rec_tgt_trust_memory(
            rec_tgt_write, rec_pad_mask, update_mask=rec_has_click)

        src2rec_sim_numer = (src2rec * rec_his_emb).sum(dim=-1)
        src2rec_sim_denom = src2rec.norm(dim=-1) * rec_his_emb.norm(dim=-1) + 1e-8
        src2rec_sim = src2rec_sim_numer / src2rec_sim_denom
        src2rec_sim = src2rec_sim.masked_fill(~rec_has_click, 0.0)

        rec_mem_write = src2rec * src2rec_sim.unsqueeze(-1)
        rec_mem_write = torch.where(
            rec_pad_mask.unsqueeze(-1),
            torch.zeros_like(rec_mem_write),
            rec_mem_write)
        rec_memory_trust_bias, _ = self.rec_memory_trust_memory(
            rec_mem_write, rec_pad_mask, update_mask=rec_has_click)

        src_memory_trust_bias = None
        src_tgt_trust_bias = None
        memory_state = None
        self._last_trust_bias = None
        if domain == 'src':
            query_emb, click_item_mask, q_click_item_emb = q_i_align_used
            click_item_sum = torch.sum(q_click_item_emb *
                                       click_item_mask.unsqueeze(-1),
                                       dim=-2)
            click_count = click_item_mask.sum(-1,
                                              keepdim=True).clamp(min=1.0)
            mean_click_item_emb = click_item_sum / click_count
            has_click = click_item_mask.sum(-1) > 0

            src_selector = (all_his_type == 2)
            src_len = src2src.size(1)
            src_mean_click = torch.masked_select(
                mean_click_item_emb, src_selector.unsqueeze(-1)).reshape(
                    (mean_click_item_emb.size(0), src_len,
                     mean_click_item_emb.size(-1)))
            src_has_click = torch.masked_select(has_click,
                                                src_selector).reshape(
                                                    (has_click.size(0),
                                                     src_len))
            # src 分支 memory gate：rec2src 与搜索点击均值一致性
            sim_numer = (rec2src * src_mean_click).sum(dim=-1)
            sim_denom = rec2src.norm(dim=-1) * src_mean_click.norm(dim=-1) + 1e-8
            sim = sim_numer / sim_denom
            sim = sim.masked_fill(~src_has_click, 0.0)

            write_signal = rec2src * sim.unsqueeze(-1)
            write_signal = torch.where(
                src_pad_mask.unsqueeze(-1),
                torch.zeros_like(write_signal),
                write_signal)

            src_memory_trust_bias, memory_state = self.src_memory_trust_memory(write_signal,
                                                                               src_pad_mask,
                                                                               update_mask=src_has_click)
            self._last_trust_bias = src_memory_trust_bias

            # src 分支 tgt gate：src2src 与搜索点击均值一致性
            sim_src_numer = (src2src * src_mean_click).sum(dim=-1)
            sim_src_denom = src2src.norm(dim=-1) * src_mean_click.norm(dim=-1) + 1e-8
            sim_src = sim_src_numer / sim_src_denom
            sim_src = sim_src.masked_fill(~src_has_click, 0.0)

            src_write_signal = src2src * sim_src.unsqueeze(-1)
            src_write_signal = torch.where(
                src_pad_mask.unsqueeze(-1),
                torch.zeros_like(src_write_signal),
                src_write_signal)

            src_tgt_trust_bias, _ = self.src_tgt_trust_memory(
                src_write_signal, src_pad_mask, update_mask=src_has_click)

        src2rec_pad = (src2rec.abs().sum(dim=-1) == 0)
        rec2src_pad = (rec2src.abs().sum(dim=-1) == 0)
        rec2rec_pad = (rec2rec.abs().sum(dim=-1) == 0)
        src2src_pad = (src2src.abs().sum(dim=-1) == 0)

        rec_fusion_decoded = self.rec_cross_fusion(
            tgt=rec2rec,
            memory=src2rec,
            tgt_key_padding_mask=rec_pad_mask,
            memory_key_padding_mask=self.match_mask_to_tensor(src2rec_pad, src2rec),
            memory_scale=rec_memory_trust_bias,
            tgt_scale=rec_tgt_trust_bias)
        self._check_finite("rec_fusion_decoded", rec_fusion_decoded)

        src_fusion_decoded = self.src_cross_fusion(
            tgt=src2src,
            memory=rec2src,
            tgt_key_padding_mask=src_pad_mask,
            memory_key_padding_mask=self.match_mask_to_tensor(rec2src_pad, rec2src),
            memory_scale=src_memory_trust_bias,
            tgt_scale=src_tgt_trust_bias)
        self._check_finite("src_fusion_decoded", src_fusion_decoded)

        if self.memory_log and src_memory_trust_bias is not None:
            self._memory_log_counter += 1
            if self._memory_log_counter % max(1, self.memory_log_interval) == 0:
                tb = src_memory_trust_bias
                print("[MemoryLog]",
                      f"step={self._memory_log_counter}",
                      f"bias_mean={tb.mean().item():.4f}",
                      f"bias_min={tb.min().item():.4f}",
                      f"bias_max={tb.max().item():.4f}",
                      f"w_norm={self.src_memory_trust_memory.trust_proj.weight.norm().item():.4f}",
                      f"mem_norm={memory_state.norm(dim=-1).mean().item():.4f}" if memory_state is not None else "")

        his_cl_used = [
            src2rec, rec2rec, self.match_mask_to_tensor(rec_pad_mask, rec2rec),
            rec2src, src2src, self.match_mask_to_tensor(src_pad_mask, src2src)
        ]

        rec_his_mask, src_his_mask = rec_pad_mask, src_pad_mask
        if items_emb.dim() == 3:
            feature_list = [
                rec_fusion_decoded, rec_pad_mask, src_fusion_decoded,
                src_pad_mask, user_emb
            ]
            repeat_feature_list, items_emb = self.repeat_feat(
                feature_list, items_emb)
            rec_fusion_decoded, rec_his_mask,\
                src_fusion_decoded, src_his_mask,\
                user_emb = repeat_feature_list

        rec_fusion = self.rec_his_attn_pooling(rec_fusion_decoded, items_emb,
                                               rec_his_mask)
        src_fusion = self.src_his_attn_pooling(src_fusion_decoded, items_emb,
                                               src_his_mask)
        self._check_finite("rec_fusion", rec_fusion)
        self._check_finite("src_fusion", src_fusion)

        user_feats = [rec_fusion, src_fusion, user_emb]

        return user_feats, q_i_align_used, his_cl_used

    def inter_pred(self, user_feats, item_emb, domain, query_emb=None):
        assert domain in ["rec", "src"]

        rec_interest, src_interest, user_emb = user_feats

        if domain == "rec":
            item_emb = item_emb.reshape(-1, item_emb.size(-1))

            concat = torch.cat([
                rec_interest, src_interest, item_emb, user_emb,
                self.rec_query.expand(item_emb.shape[0], -1)
            ], -1)
            if not torch.isfinite(concat).all():
                print("[inter_pred-rec] concat 存在 NaN/Inf")
            output = self.ple_layer(concat)[0]
            fc_out = self.rec_fc_layer(output)
            return fc_out

        elif domain == "src":
            if item_emb.dim() == 3:
                [query_emb], item_emb = self.repeat_feat([query_emb], item_emb)

            concat = torch.cat(
                [rec_interest, src_interest, item_emb, user_emb, query_emb], -1)
            output = self.ple_layer(concat)[1]
            fc_out = self.src_fc_layer(output)
            return fc_out

    def rec_loss(self, inputs):
        user, all_his, all_his_type, pos_item, neg_items = inputs[
            'user'], inputs['all_his'], inputs['all_his_type'], inputs[
                'item'], inputs['neg_items']

        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        self._debug_tensor_range("rec_loss_items", items)
        items_emb = self.session_embedding.get_item_emb(items)
        self._check_finite("rec_loss_items_emb", items_emb)
        batch_size = items_emb.size(0)

        user_feats, q_i_align_used, his_cl_used = self.forward(user,
                                                               all_his,
                                                               all_his_type,
                                                               items_emb,
                                                               domain='rec')

        logits = self.inter_pred(user_feats, items_emb, domain="rec").reshape(
            (batch_size, -1))
        labels = torch.zeros_like(logits, dtype=torch.float32)
        labels[:, 0] = 1.0

        logits = logits.reshape((-1, ))
        labels = labels.reshape((-1, ))



        total_loss = self.loss_fn(logits, labels)
        loss_dict = {}
        loss_dict['click_loss'] = total_loss.clone()

        if self.q_i_cl_weight > 0:
            align_neg_item, align_neg_query = inputs['align_neg_item'], inputs[
                'align_neg_query']
            query_emb, click_item_mask, q_click_item_emb = q_i_align_used

            align_neg_items_emb = self.session_embedding.get_item_emb(
                align_neg_item)
            align_neg_querys_emb = self.session_embedding.get_query_emb(
                align_neg_query)
            align_loss = self.feature_alignment(
                [align_neg_items_emb, align_neg_querys_emb], query_emb,
                click_item_mask, q_click_item_emb)
            loss_dict['q_i_cl_loss'] = align_loss.clone()

            total_loss += self.q_i_cl_weight * align_loss

        if self.his_cl_weight > 0:
            src2rec, rec2rec, rec_his_mask,\
                rec2src, src2src, src_his_mask = his_cl_used
            rec_his_cl_loss = self.rec_his_cl(src2rec, rec2rec, rec_his_mask)

            src_his_cl_loss = self.src_his_cl(rec2src, src2src, src_his_mask)

            his_cl_loss = rec_his_cl_loss + src_his_cl_loss
            loss_dict['his_cl_loss'] = his_cl_loss.clone()

            total_loss += self.his_cl_weight * his_cl_loss

        loss_dict['total_loss'] = total_loss

        return loss_dict

    def rec_predict(self, inputs):
        user, all_his, all_his_type, pos_item, neg_items = inputs[
            'user'], inputs['all_his'], inputs['all_his_type'], inputs[
                'item'], inputs['neg_items']

        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        self._debug_tensor_range("rec_predict_items", items)
        items_emb = self.session_embedding.get_item_emb(items)
        self._check_finite("rec_predict_items_emb", items_emb)
        batch_size = items_emb.size(0)

        user_feats, q_i_align_used, his_cl_used = self.forward(user,
                                                               all_his,
                                                               all_his_type,
                                                               items_emb,
                                                               domain='rec')

        logits = self.inter_pred(user_feats, items_emb, domain="rec").reshape(
            (batch_size, -1))
        return logits

    def src_loss(self, inputs):
        user, all_his, all_his_type, pos_item, neg_items = inputs[
            'user'], inputs['all_his'], inputs['all_his_type'], inputs[
                'item'], inputs['neg_items']

        query = inputs['query']
        query_emb = self.session_embedding.get_query_emb(query)

        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        self._debug_tensor_range("src_loss_items", items)
        items_emb = self.session_embedding.get_item_emb(items)
        self._check_finite("src_loss_items_emb", items_emb)
        batch_size = items_emb.size(0)

        user_feats, q_i_align_used, his_cl_used = self.forward(user,
                                                               all_his,
                                                               all_his_type,
                                                               items_emb,
                                                               domain='src')

        logits = self.inter_pred(user_feats,
                                 items_emb,
                                 domain="src",
                                 query_emb=query_emb).reshape((batch_size, -1))
        labels = torch.zeros_like(logits, dtype=torch.float32)
        labels[:, 0] = 1.0

        logits = logits.reshape((-1, ))
        labels = labels.reshape((-1, ))

        total_loss = self.loss_fn(logits, labels)
        loss_dict = {}
        loss_dict['click_loss'] = total_loss.clone()

        if self.q_i_cl_weight > 0:
            align_neg_item, align_neg_query = inputs['align_neg_item'], inputs[
                'align_neg_query']
            query_emb, click_item_mask, q_click_item_emb = q_i_align_used

            align_neg_items_emb = self.session_embedding.get_item_emb(
                align_neg_item)
            align_neg_querys_emb = self.session_embedding.get_query_emb(
                align_neg_query)
            align_loss = self.feature_alignment(
                [align_neg_items_emb, align_neg_querys_emb], query_emb,
                click_item_mask, q_click_item_emb)
            loss_dict['q_i_cl_loss'] = align_loss.clone()

            total_loss += self.q_i_cl_weight * align_loss

        if self.his_cl_weight > 0:
            src2rec, rec2rec, rec_his_mask,\
                rec2src, src2src, src_his_mask = his_cl_used

            rec_his_cl_loss = self.rec_his_cl(src2rec, rec2rec, rec_his_mask)

            src_his_cl_loss = self.src_his_cl(rec2src, src2src, src_his_mask)

            his_cl_loss = rec_his_cl_loss + src_his_cl_loss
            loss_dict['his_cl_loss'] = his_cl_loss.clone()

            total_loss += self.his_cl_weight * his_cl_loss

        loss_dict['total_loss'] = total_loss

        return loss_dict

    def src_predict(self, inputs):
        user, all_his, all_his_type, pos_item, neg_items = inputs[
            'user'], inputs['all_his'], inputs['all_his_type'], inputs[
                'item'], inputs['neg_items']

        query = inputs['query']
        query_emb = self.session_embedding.get_query_emb(query)

        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        self._debug_tensor_range("src_predict_items", items)
        items_emb = self.session_embedding.get_item_emb(items)
        self._check_finite("src_predict_items_emb", items_emb)
        batch_size = items_emb.size(0)

        user_feats, q_i_align_used, his_cl_used = self.forward(user,
                                                               all_his,
                                                               all_his_type,
                                                               items_emb,
                                                               domain='src')

        logits = self.inter_pred(user_feats,
                                 items_emb,
                                 domain="src",
                                 query_emb=query_emb).reshape((batch_size, -1))
        return logits


class Target_Attention(nn.Module):
    def __init__(self, hid_dim1, hid_dim2):
        super().__init__()

        self.W = nn.Parameter(torch.randn((1, hid_dim1, hid_dim2)))
        nn.init.xavier_normal_(self.W)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, seq_emb, target, mask):
        score = torch.matmul(seq_emb, self.W)
        score = torch.matmul(score, target.unsqueeze(-1))

        all_score = score.masked_fill(mask.unsqueeze(-1), torch.tensor(-1e16))
        all_weight = self.softmax(all_score.transpose(-2, -1))
        all_vec = torch.matmul(all_weight, seq_emb).squeeze(1)

        return all_vec


class TransAlign(nn.Module):
    def __init__(self, batch_size, hidden_dim, device, infoNCE_temp) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.device = device

        self.infoNCE_temp = nn.Parameter(torch.ones([]) * infoNCE_temp)
        self.weight_matrix = nn.Parameter(torch.randn(
            (hidden_dim, hidden_dim)))
        nn.init.xavier_normal_(self.weight_matrix)

        self.cl_loss_func = nn.CrossEntropyLoss()
        self.mask_default = self.mask_correlated_samples(self.batch_size)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool, device=self.device)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, same_his: torch.Tensor, diff_his: torch.Tensor,
                his_mask: torch.Tensor):
        same_his_emb = same_his.masked_fill(his_mask.unsqueeze(2), 0)
        same_his_sum = same_his_emb.sum(dim=1)
        same_his_mean = same_his_sum / \
            (~his_mask).sum(dim=1, keepdim=True)

        diff_his_emb = diff_his.masked_fill(his_mask.unsqueeze(2), 0)
        diff_his_sum = diff_his_emb.sum(dim=1)
        diff_his_mean = diff_his_sum / \
            (~his_mask).sum(dim=1, keepdim=True)

        batch_size = same_his_mean.size(0)
        N = 2 * batch_size

        z = torch.cat([same_his_mean.squeeze(),
                       diff_his_mean.squeeze()],
                      dim=0)
        sim = torch.mm(torch.mm(z, self.weight_matrix), z.T)
        sim = torch.tanh(sim) / self.infoNCE_temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        info_nce_loss = self.cl_loss_func(logits, labels)

        return info_nce_loss


class SlowTrustMemory(nn.Module):
    def __init__(self,
                 dim: int,
                 epsilon: float = 0.01,
                 clamp_min: float = 0.8,
                 clamp_max: float = 1.2):
        super().__init__()
        self.epsilon = epsilon
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.trust_proj = nn.Linear(dim, 1, bias=False)
        nn.init.zeros_(self.trust_proj.weight)

    def forward(self,
                write_signal: torch.Tensor,
                pad_mask: torch.Tensor,
                update_mask: torch.Tensor = None):
        """
        write_signal: (batch, T, dim)
        pad_mask: (batch, T) with True indicating padding
        update_mask: optional (batch, T) bool, True to update at t
        """
        if write_signal.numel() == 0:
            return write_signal.new_zeros((write_signal.size(0), 0)), None

        batch, seq_len, dim = write_signal.shape
        memory = write_signal.new_zeros((batch, dim))
        biases = []
        for t in range(seq_len):
            update_gate = (~pad_mask[:, t]).float().unsqueeze(-1)
            if update_mask is not None:
                update_gate = update_gate * update_mask[:, t].float().unsqueeze(
                    -1)
            memory = (1 - self.epsilon * update_gate) * memory + \
                self.epsilon * write_signal[:, t, :] * update_gate
            bias = 1 + torch.tanh(self.trust_proj(memory)).squeeze(-1)
            biases.append(bias)

        bias_stack = torch.stack(biases, dim=1)
        bias_stack = bias_stack.masked_fill(pad_mask, 1.0)
        bias_stack = torch.clamp(bias_stack,
                                 min=self.clamp_min,
                                 max=self.clamp_max)
        return bias_stack, memory


class MemoryTransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 batch_first: bool = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model,
                                               nhead,
                                               dropout=dropout,
                                               batch_first=batch_first)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout,
                                                    batch_first=batch_first)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None,
                memory_scale: torch.Tensor = None,
                tgt_scale: torch.Tensor = None):
        # Soft gate: convex blend to avoid hard zeroing (scale in [gate_floor, +inf))
        gate_floor = getattr(self, 'gate_floor', 0.85)
        def _blend(x, scale):
            if scale is None:
                return x
            s = torch.clamp(scale, min=gate_floor).unsqueeze(-1)
            return s * x + (1 - s) * x.detach()

        gated_tgt = _blend(tgt, tgt_scale)
        tgt2 = self.self_attn(gated_tgt,
                              gated_tgt,
                              gated_tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 使用 trust bias 缩放 memory（来自 rec2src），而不是缩放当前查询序列
        scaled_memory = _blend(memory, memory_scale)
        tgt2 = self.multihead_attn(tgt,
                                   scaled_memory,
                                   scaled_memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class MemoryTransformerDecoder(nn.Module):
    def __init__(self,
                 decoder_layer: MemoryTransformerDecoderLayer,
                 num_layers: int,
                 norm: nn.Module = None):
        super().__init__()
        self.layers = nn.ModuleList(
            [deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None,
                memory_scale: torch.Tensor = None,
                tgt_scale: torch.Tensor = None):
        output = tgt

        for mod in self.layers:
            output = mod(output,
                         memory,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         memory_scale=memory_scale,
                         tgt_scale=tgt_scale)

        if self.norm is not None:
            output = self.norm(output)

        return output


class Transformer(nn.Module):
    def __init__(self, emb_size, num_heads, num_layers, dropout) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.transformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=emb_size,
            dropout=dropout,
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformerEncoderLayer, num_layers=num_layers)

    def forward(self,
                his_emb: torch.Tensor,
                src_key_padding_mask: torch.Tensor,
                src_mask: torch.Tensor = None):
        if not torch.isfinite(his_emb).all():
            print("[Transformer] 输入 his_emb 存在 NaN/Inf")
        if src_mask is not None and not torch.isfinite(src_mask.float()).all():
            print("[Transformer] 输入 src_mask 存在 NaN/Inf")

        if src_mask is not None:
            src_mask_expand = src_mask.unsqueeze(1).expand(
                (-1, self.num_heads, -1, -1)).reshape(
                    (-1, his_emb.size(1), his_emb.size(1)))
            his_encoded = self.transformer_encoder(
                src=his_emb,
                src_key_padding_mask=src_key_padding_mask,
                mask=src_mask_expand)
        else:
            his_encoded = self.transformer_encoder(
                src=his_emb, src_key_padding_mask=src_key_padding_mask)
        if not torch.isfinite(his_encoded).all():
            print("[Transformer] 输出 his_encoded 存在 NaN/Inf")

        return his_encoded

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import math
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
        parser.add_argument('--rec_use_src_interest', action='store_true',
                            help='Whether to feed src_interest into rec prediction branch')
        parser.add_argument('--intent_num', type=int, default=4)
        parser.add_argument('--intent_heads', type=int, default=2)
        parser.add_argument('--intent_temp', type=float, default=1.0)
        parser.add_argument('--intent_dropout', type=float, default=0.1)
        parser.add_argument('--transition_dynamic_hidden', type=int, default=64)
        parser.add_argument('--intent_diag', action='store_true')
        parser.add_argument('--intent_diag_interval', type=int, default=200)
        parser.add_argument('--intent_entropy_floor', type=float, default=0.45)
        parser.add_argument('--intent_top1_ceiling', type=float, default=0.75)
        parser.add_argument('--intent_proto_sim_ceiling', type=float, default=0.85)
        parser.add_argument('--transition_entropy_floor', type=float, default=0.35)
        parser.add_argument('--transition_peak_ceiling', type=float, default=0.85)
        parser.add_argument('--transition_cycle_ceiling', type=float, default=0.90)

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

        # Learnable mix between pure rec path and src->rec cross fusion
        self.rec_src_mix = nn.Parameter(torch.tensor(0.5))

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
        self.rec_use_src_interest = args.rec_use_src_interest
        self.intent_num = args.intent_num
        self.intent_temp = args.intent_temp
        # 双分支信任记忆：src 用于 memory=rec2src / tgt=src2src，rec 用于 memory=src2rec / tgt=rec2rec
        self.src_memory_trust_memory = TrustMemory(dim=self.item_size,
                                                       epsilon=self.memory_eps,
                                                       clamp_min=args.memory_bias_min,
                                                       clamp_max=args.memory_bias_max)
        self.src_tgt_trust_memory = TrustMemory(dim=self.item_size,
                                                    epsilon=self.memory_eps,
                                                    clamp_min=args.memory_bias_min,
                                                    clamp_max=args.memory_bias_max)
        self.rec_memory_trust_memory = TrustMemory(dim=self.item_size,
                                                       epsilon=self.memory_eps,
                                                       clamp_min=args.memory_bias_min,
                                                       clamp_max=args.memory_bias_max)
        self.rec_tgt_trust_memory = TrustMemory(dim=self.item_size,
                                                    epsilon=self.memory_eps,
                                                    clamp_min=args.memory_bias_min,
                                                    clamp_max=args.memory_bias_max)
        self.rec_intent_discovery = LatentIntentDiscovery(
            emb_dim=self.item_size,
            num_intents=self.intent_num,
            num_heads=args.intent_heads,
            dropout=args.intent_dropout)
        self.src_intent_discovery = LatentIntentDiscovery(
            emb_dim=self.item_size,
            num_intents=self.intent_num,
            num_heads=args.intent_heads,
            dropout=args.intent_dropout)
        self.intent_transition_graph = IntentTransitionGraph(
            emb_dim=self.item_size,
            num_intents=self.intent_num,
            hidden_dim=args.transition_dynamic_hidden)
        self.memory_log = args.memory_log
        self.memory_log_interval = args.memory_log_interval
        self.intent_diag = args.intent_diag
        self.intent_diag_interval = args.intent_diag_interval
        self.intent_entropy_floor = args.intent_entropy_floor
        self.intent_top1_ceiling = args.intent_top1_ceiling
        self.intent_proto_sim_ceiling = args.intent_proto_sim_ceiling
        self.transition_entropy_floor = args.transition_entropy_floor
        self.transition_peak_ceiling = args.transition_peak_ceiling
        self.transition_cycle_ceiling = args.transition_cycle_ceiling
        self._memory_log_counter = 0
        self._last_trust_bias = None
        self._intent_diag_counter = 0
        self._last_intent_diag = None
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

    def _safe_cosine(self, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
        numer = (a * b).sum(dim=-1)
        denom = a.norm(dim=-1) * b.norm(dim=-1) + eps
        return numer / denom

    def _intent_soft_assign(self,
                            behavior: torch.Tensor,
                            intents: torch.Tensor,
                            pad_mask: torch.Tensor = None):
        logits = torch.einsum("btd,bkd->btk", behavior, intents)
        logits = logits / (behavior.size(-1) ** 0.5 * max(self.intent_temp, 1e-6))
        assign = torch.softmax(logits, dim=-1)
        if pad_mask is not None:
            assign = assign.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return assign

    def _normalized_entropy(self,
                            probs: torch.Tensor,
                            dim: int = -1,
                            eps: float = 1e-8):
        probs = probs.clamp_min(eps)
        entropy = -(probs * probs.log()).sum(dim=dim)
        support = probs.size(dim)
        if support <= 1:
            return torch.zeros_like(entropy)
        return entropy / math.log(support)

    @torch.no_grad()
    def _collect_intent_diagnostics(self,
                                    rec_assign: torch.Tensor,
                                    src_assign: torch.Tensor,
                                    rec_intents: torch.Tensor,
                                    src_intents: torch.Tensor,
                                    r2s_probs: torch.Tensor,
                                    s2r_probs: torch.Tensor,
                                    rec_pad_mask: torch.Tensor,
                                    src_pad_mask: torch.Tensor):
        diag = {}
        valid_groups = [
            ("rec", rec_assign, rec_intents, rec_pad_mask),
            ("src", src_assign, src_intents, src_pad_mask)
        ]
        for name, assign, intents, pad_mask in valid_groups:
            valid_mask = (~pad_mask).unsqueeze(-1).float()
            valid_count = valid_mask.sum().clamp(min=1.0)
            entropy = self._normalized_entropy(assign, dim=-1)
            entropy = (entropy * valid_mask.squeeze(-1)).sum() / valid_count
            top1 = (assign.max(dim=-1).values * valid_mask.squeeze(-1)).sum() / valid_count
            usage = (assign * valid_mask).sum(dim=(0, 1)) / valid_count
            proto = F.normalize(intents.mean(dim=0), dim=-1)
            proto_sim = torch.matmul(proto, proto.transpose(0, 1))
            proto_mask = ~torch.eye(proto_sim.size(0),
                                    dtype=torch.bool,
                                    device=proto_sim.device)
            if proto_mask.any():
                proto_sim_mean = proto_sim.masked_select(proto_mask).mean()
            else:
                proto_sim_mean = torch.tensor(0.0, device=proto_sim.device)
            usage_entropy = self._normalized_entropy(usage.unsqueeze(0), dim=-1).squeeze(0)
            effective = torch.exp(usage_entropy * math.log(max(usage.numel(), 2)))
            diag[f"{name}_entropy"] = entropy.item()
            diag[f"{name}_top1"] = top1.item()
            diag[f"{name}_effective"] = effective.item()
            diag[f"{name}_usage_peak"] = usage.max().item()
            diag[f"{name}_proto_sim"] = proto_sim_mean.item()
            diag[f"{name}_collapse"] = (
                entropy.item() < self.intent_entropy_floor or
                top1.item() > self.intent_top1_ceiling or
                proto_sim_mean.item() > self.intent_proto_sim_ceiling
            )

        round_trip_groups = [
            ("rec", torch.bmm(r2s_probs, s2r_probs)),
            ("src", torch.bmm(s2r_probs, r2s_probs))
        ]
        for name, transition in round_trip_groups:
            row_entropy = self._normalized_entropy(transition, dim=-1).mean()
            row_peak = transition.max(dim=-1).values.mean()
            trace_mean = transition.diagonal(dim1=-2, dim2=-1).mean()
            cycle_gap = (row_peak - trace_mean).clamp(min=0.0)
            offdiag_mass = (1.0 - trace_mean).clamp(min=0.0, max=1.0)
            diag[f"{name}_transition_entropy"] = row_entropy.item()
            diag[f"{name}_transition_peak"] = row_peak.item()
            diag[f"{name}_transition_trace"] = trace_mean.item()
            diag[f"{name}_cycle_gap"] = cycle_gap.item()
            diag[f"{name}_transition_offdiag"] = offdiag_mass.item()
            diag[f"{name}_non_convergent_risk"] = (
                row_entropy.item() < self.transition_entropy_floor and
                offdiag_mass.item() > self.transition_cycle_ceiling
            )
            diag[f"{name}_transition_collapse"] = (
                row_entropy.item() < self.transition_entropy_floor or
                row_peak.item() > self.transition_peak_ceiling
            )
        return diag

    def _log_intent_diagnostics(self, diag):
        flags = []
        for prefix in ("rec", "src"):
            if diag.get(f"{prefix}_collapse", False):
                flags.append(f"{prefix}_intent_collapse")
            if diag.get(f"{prefix}_transition_collapse", False):
                flags.append(f"{prefix}_transition_collapse")
            if diag.get(f"{prefix}_non_convergent_risk", False):
                flags.append(f"{prefix}_cycle_risk")
        flag_text = ",".join(flags) if flags else "ok"
        print("[IntentDiag]",
              f"step={self._intent_diag_counter}",
              f"flags={flag_text}",
              f"rec_H={diag['rec_entropy']:.3f}",
              f"rec_top1={diag['rec_top1']:.3f}",
              f"rec_eff={diag['rec_effective']:.2f}",
              f"rec_proto={diag['rec_proto_sim']:.3f}",
              f"rec_T_H={diag['rec_transition_entropy']:.3f}",
              f"rec_T_peak={diag['rec_transition_peak']:.3f}",
              f"rec_cycle={diag['rec_cycle_gap']:.3f}",
              f"src_H={diag['src_entropy']:.3f}",
              f"src_top1={diag['src_top1']:.3f}",
              f"src_eff={diag['src_effective']:.2f}",
              f"src_proto={diag['src_proto_sim']:.3f}",
              f"src_T_H={diag['src_transition_entropy']:.3f}",
              f"src_T_peak={diag['src_transition_peak']:.3f}",
              f"src_cycle={diag['src_cycle_gap']:.3f}")

    def _current_item_anchor(self, items_emb: torch.Tensor):
        if items_emb.dim() == 3:
            # Use the first candidate as current anchor (positive item during training/eval pipeline).
            return items_emb[:, 0, :]
        return items_emb

    def _intent_routing_bias(self,
                             transition: torch.Tensor,
                             position_item: torch.Tensor,
                             src_intents: torch.Tensor,
                             tgt_intents: torch.Tensor,
                             src_to_tgt_probs: torch.Tensor,
                             pad_mask: torch.Tensor,
                             update_mask: torch.Tensor,
                             memory_module):
        valid_mask = (~pad_mask)
        if update_mask is not None:
            valid_mask = valid_mask & update_mask

        # Semantic confidence of each transition step under current history position item.
        alpha_abs = torch.abs((transition * position_item).sum(dim=-1))
        alpha_conf = torch.tanh(alpha_abs).masked_fill(~valid_mask, 0.0)

        item_tgt_logits = torch.einsum("btd,bkd->btk", position_item, tgt_intents)
        item_tgt_logits = item_tgt_logits / (position_item.size(-1) ** 0.5 * max(self.intent_temp, 1e-6))
        item_tgt_assign = torch.softmax(item_tgt_logits, dim=-1)
        src_demand = torch.einsum("btk,bmk->btm", item_tgt_assign, src_to_tgt_probs.transpose(1, 2))
        mem_src_assign = self._intent_soft_assign(transition, src_intents, pad_mask=pad_mask)

        route_match = (src_demand * mem_src_assign).sum(dim=-1)
        route_match = route_match.masked_fill(~valid_mask, 0.0)

        src_reliability_per_intent = src_to_tgt_probs.max(dim=-1).values
        src_reliability = (mem_src_assign * src_reliability_per_intent.unsqueeze(1)).sum(dim=-1)
        src_reliability = src_reliability.masked_fill(~valid_mask, 0.0)

        confidence = alpha_conf * route_match * src_reliability
        confidence = confidence.masked_fill(pad_mask, 0.0)
        confidence = confidence.clamp(min=0.0, max=1.0)

        trust_bias = memory_module.clamp_min + \
            (memory_module.clamp_max - memory_module.clamp_min) * confidence
        neutral_mask = (~pad_mask) & (~valid_mask)
        trust_bias = trust_bias.masked_fill(neutral_mask, 1.0)
        trust_bias = trust_bias.masked_fill(pad_mask, 1.0)
        trust_bias = torch.clamp(trust_bias,
                                 min=memory_module.clamp_min,
                                 max=memory_module.clamp_max)
        return trust_bias

    def _transition_confidence_bias(self,
                                    transition: torch.Tensor,
                                    item_anchor: torch.Tensor,
                                    pad_mask: torch.Tensor,
                                    update_mask: torch.Tensor,
                                    memory_module):
        valid_mask = (~pad_mask)
        if update_mask is not None:
            valid_mask = valid_mask & update_mask

        # Factor 1: absolute alpha strength between current item and transition (no softmax)
        alpha_abs = torch.abs((transition * item_anchor).sum(dim=-1))
        alpha_abs = alpha_abs.masked_fill(~valid_mask, 0.0)
        # Monotonic compression without per-sample normalization.
        alpha_conf = torch.tanh(alpha_abs)

        write_signal = transition * alpha_conf.unsqueeze(-1)
        write_signal = torch.where(pad_mask.unsqueeze(-1), torch.zeros_like(write_signal),
                                   write_signal)

        _, memory_state, memory_trace, stability = memory_module(
            write_signal,
            pad_mask,
            update_mask=update_mask,
            return_trace=True)

        # Factor 2: path membership confidence (current item belongs to cumulative memory path)
        path_sim = self._safe_cosine(item_anchor, memory_trace)
        path_conf = ((path_sim + 1.0) * 0.5).masked_fill(~valid_mask, 0.0)

        # Factor 3: memory/path stability confidence
        stability = stability.masked_fill(~valid_mask, 0.0)

        confidence = alpha_conf * path_conf * stability
        confidence = confidence.masked_fill(pad_mask, 0.0)

        trust_bias = memory_module.clamp_min + \
            (memory_module.clamp_max - memory_module.clamp_min) * confidence
        # Keep non-updated (no evidence) non-pad steps neutral instead of punitive.
        neutral_mask = (~pad_mask) & (~valid_mask)
        trust_bias = trust_bias.masked_fill(neutral_mask, 1.0)
        trust_bias = trust_bias.masked_fill(pad_mask, 1.0)
        trust_bias = torch.clamp(trust_bias,
                                 min=memory_module.clamp_min,
                                 max=memory_module.clamp_max)
        return trust_bias, memory_state

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

        rec_intents = self.rec_intent_discovery(rec2rec, rec_pad_mask)
        src_intents = self.src_intent_discovery(src2src, src_pad_mask)
        r2s_probs, s2r_probs = self.intent_transition_graph(rec_intents, src_intents)
        if self.intent_diag:
            with torch.no_grad():
                rec_assign = self._intent_soft_assign(rec2rec.detach(),
                                                      rec_intents.detach(),
                                                      pad_mask=rec_pad_mask)
                src_assign = self._intent_soft_assign(src2src.detach(),
                                                      src_intents.detach(),
                                                      pad_mask=src_pad_mask)
                self._last_intent_diag = self._collect_intent_diagnostics(
                    rec_assign=rec_assign,
                    src_assign=src_assign,
                    rec_intents=rec_intents.detach(),
                    src_intents=src_intents.detach(),
                    r2s_probs=r2s_probs.detach(),
                    s2r_probs=s2r_probs.detach(),
                    rec_pad_mask=rec_pad_mask,
                    src_pad_mask=src_pad_mask)
                self._intent_diag_counter += 1
                if self._intent_diag_counter % max(1, self.intent_diag_interval) == 0:
                    self._log_intent_diagnostics(self._last_intent_diag)
        src_source_available = (~src_pad_mask).any(dim=1, keepdim=True).expand_as(rec_pad_mask)
        rec_source_available = (~rec_pad_mask).any(dim=1, keepdim=True).expand_as(src_pad_mask)

        # rec 分支 confidence gate：alpha强度 + 路径归属 + 路径稳定性
        rec_has_click = ~rec_pad_mask
        rec_tgt_trust_bias, _ = self._transition_confidence_bias(
            transition=rec2rec,
            item_anchor=rec_his_emb,
            pad_mask=rec_pad_mask,
            update_mask=rec_has_click,
            memory_module=self.rec_tgt_trust_memory)

        rec_memory_trust_bias = self._intent_routing_bias(
            transition=src2rec,
            position_item=rec_his_emb,
            src_intents=src_intents,
            tgt_intents=rec_intents,
            src_to_tgt_probs=s2r_probs,
            pad_mask=rec_pad_mask,
            update_mask=src_source_available,
            memory_module=self.rec_memory_trust_memory)

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
        # src 分支 memory gate：统一 confidence（三因子）
        src_memory_trust_bias = self._intent_routing_bias(
            transition=rec2src,
            position_item=src_mean_click,
            src_intents=rec_intents,
            tgt_intents=src_intents,
            src_to_tgt_probs=r2s_probs,
            pad_mask=src_pad_mask,
            update_mask=rec_source_available,
            memory_module=self.src_memory_trust_memory)
        memory_state = rec2src.masked_fill(src_pad_mask.unsqueeze(-1), 0.0).mean(dim=1)
        self._last_trust_bias = src_memory_trust_bias

        # src 分支 tgt gate：统一 confidence（三因子）
        src_tgt_trust_bias, _ = self._transition_confidence_bias(
            transition=src2src,
            item_anchor=src_mean_click,
            pad_mask=src_pad_mask,
            update_mask=src_has_click,
            memory_module=self.src_tgt_trust_memory)

        src2rec_pad = (src2rec.abs().sum(dim=-1) == 0)
        rec2src_pad = (rec2src.abs().sum(dim=-1) == 0)
        rec2rec_pad = (rec2rec.abs().sum(dim=-1) == 0)
        src2src_pad = (src2src.abs().sum(dim=-1) == 0)

        # Cross path: src -> rec
        rec_cross = self.rec_cross_fusion(
            tgt=rec2rec,
            memory=src2rec,
            tgt_key_padding_mask=rec_pad_mask,
            memory_key_padding_mask=self.match_mask_to_tensor(src2rec_pad, src2rec),
            memory_scale=rec_memory_trust_bias,
            tgt_scale=rec_tgt_trust_bias)
        # Self path: pure rec history (no cross) to avoid competition
        rec_self = rec2rec
        mix = torch.sigmoid(self.rec_src_mix)
        rec_fusion_decoded = mix * rec_cross + (1 - mix) * rec_self
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

            # 控制搜索兴趣对推荐的影响：可选择完全去除或仅阻断梯度
            if self.rec_use_src_interest:
                src_interest = src_interest
            else:
                src_interest = torch.zeros_like(src_interest)

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


class LatentIntentDiscovery(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 num_intents: int,
                 num_heads: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        if emb_dim % num_heads != 0:
            num_heads = 1
        self.intent_slots = nn.Parameter(torch.randn(num_intents, emb_dim))
        nn.init.xavier_normal_(self.intent_slots)
        self.slot_attention = nn.MultiheadAttention(embed_dim=emb_dim,
                                                    num_heads=num_heads,
                                                    dropout=dropout,
                                                    batch_first=True)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, behavior_seq: torch.Tensor, pad_mask: torch.Tensor):
        batch_size = behavior_seq.size(0)
        slots = self.intent_slots.unsqueeze(0).expand(batch_size, -1, -1)
        intents, _ = self.slot_attention(query=slots,
                                         key=behavior_seq,
                                         value=behavior_seq,
                                         key_padding_mask=pad_mask)
        intents = self.norm(intents + slots)
        return intents


class IntentTransitionGraph(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 num_intents: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.num_intents = num_intents
        self.r2s_global = nn.Parameter(torch.zeros(num_intents, num_intents))
        self.s2r_global = nn.Parameter(torch.zeros(num_intents, num_intents))
        self.r2s_dynamic = nn.Sequential(nn.Linear(2 * emb_dim, hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim,
                                                   num_intents * num_intents))
        self.s2r_dynamic = nn.Sequential(nn.Linear(2 * emb_dim, hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim,
                                                   num_intents * num_intents))

    def forward(self, rec_intents: torch.Tensor, src_intents: torch.Tensor):
        rec_summary = rec_intents.mean(dim=1)
        src_summary = src_intents.mean(dim=1)
        user_ctx = torch.cat([rec_summary, src_summary], dim=-1)

        r2s_delta = self.r2s_dynamic(user_ctx).reshape(-1,
                                                       self.num_intents,
                                                       self.num_intents)
        s2r_delta = self.s2r_dynamic(user_ctx).reshape(-1,
                                                       self.num_intents,
                                                       self.num_intents)

        r2s_logits = self.r2s_global.unsqueeze(0) + r2s_delta
        s2r_logits = self.s2r_global.unsqueeze(0) + s2r_delta
        r2s_probs = torch.softmax(r2s_logits, dim=-1)
        s2r_probs = torch.softmax(s2r_logits, dim=-1)
        return r2s_probs, s2r_probs


class TrustMemory(nn.Module):
    def __init__(self,
                 dim: int,
                 epsilon: float = 0.01,
                 clamp_min: float = 0.8,
                 clamp_max: float = 1.2):
        super().__init__()
        self.epsilon = epsilon
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        # Adaptive update rate with learnable base and dynamic factors.
        self.update_proj = nn.Linear(dim, 1, bias=True)
        nn.init.zeros_(self.update_proj.weight)
        nn.init.zeros_(self.update_proj.bias)
        self.base_mult = nn.Parameter(torch.tensor(20.0))
        self.dynamic_mult = nn.Parameter(torch.tensor(10.0))

    def forward(self,
                write_signal: torch.Tensor,
                pad_mask: torch.Tensor,
                update_mask: torch.Tensor = None,
                return_trace: bool = False):
        """
        write_signal: (batch, T, dim)
        pad_mask: (batch, T) with True indicating padding
        update_mask: optional (batch, T) bool, True to update at t
        """
        if write_signal.numel() == 0:
            empty_bias = write_signal.new_zeros((write_signal.size(0), 0))
            if not return_trace:
                return empty_bias, None
            empty_trace = write_signal.new_zeros((write_signal.size(0), 0,
                                                  write_signal.size(-1)))
            return empty_bias, None, empty_trace, empty_bias

        valid_gate = (~pad_mask).float()
        if update_mask is not None:
            valid_gate = valid_gate * update_mask.float()

        # Adaptive step size (larger than the old fixed epsilon).
        adaptive_signal = torch.sigmoid(self.update_proj(write_signal)).squeeze(-1)
        base = F.softplus(self.base_mult)
        dynamic = F.softplus(self.dynamic_mult)
        eff_eps = self.epsilon * (base + dynamic * adaptive_signal)
        eff_eps = torch.clamp(eff_eps, min=1e-6, max=0.95)

        gate = eff_eps * valid_gate
        a = (1.0 - gate).unsqueeze(-1)  # (B, T, 1)
        b = gate.unsqueeze(-1) * write_signal  # (B, T, D)

        mem = write_signal.new_zeros((write_signal.size(0), write_signal.size(-1)))
        traces = []
        for t in range(write_signal.size(1)):
            mem = a[:, t, :] * mem + b[:, t, :]
            traces.append(mem)
        memory_trace = torch.stack(traces, dim=1)
        memory = memory_trace[:, -1, :]

        bias_stack = write_signal.new_ones((write_signal.size(0), write_signal.size(1)))
        bias_stack = bias_stack.masked_fill(pad_mask, 1.0)
        if not return_trace:
            return bias_stack, memory

        # Full-path stability: deterministic running accumulation (no cumsum).
        prev_memory = torch.zeros_like(memory_trace[:, 0, :])
        run_count = write_signal.new_zeros((write_signal.size(0),))
        run_sum = write_signal.new_zeros((write_signal.size(0),))
        run_sum_sq = write_signal.new_zeros((write_signal.size(0),))
        stability_list = []
        for t in range(write_signal.size(1)):
            cur_memory = memory_trace[:, t, :]
            delta = (cur_memory - prev_memory).norm(dim=-1)
            base_norm = prev_memory.norm(dim=-1).clamp(min=1e-6)
            step_ratio = delta / base_norm

            valid_t = valid_gate[:, t]
            # Do not penalize first effective step of each sample.
            first_valid_t = (run_count == 0).float() * valid_t
            step_ratio_eff = torch.where(first_valid_t > 0,
                                         torch.zeros_like(step_ratio),
                                         step_ratio)
            run_count = run_count + valid_t
            run_sum = run_sum + step_ratio_eff * valid_t
            run_sum_sq = run_sum_sq + (step_ratio_eff ** 2) * valid_t

            safe_count = run_count.clamp(min=1.0)
            cum_mean = run_sum / safe_count
            cum_var = torch.clamp(run_sum_sq / safe_count - cum_mean ** 2,
                                  min=0.0)
            cum_std = torch.sqrt(cum_var + 1e-8)
            stability_t = torch.exp(-(cum_mean + cum_std))
            stability_list.append(stability_t)
            prev_memory = cur_memory

        stability_stack = torch.stack(stability_list, dim=1)
        stability_stack = stability_stack.masked_fill(pad_mask, 1.0)
        return bias_stack, memory, memory_trace, stability_stack


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
        tgt2 = self.self_attn(tgt,
                              tgt,
                              tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Additive attention bias based on trust gates
        attn_bias = None
        if tgt_scale is not None or memory_scale is not None:
            attn_bias = memory.new_zeros((memory.size(0), tgt.size(1), memory.size(1)))
            if tgt_scale is not None:
                q_scale = torch.clamp(tgt_scale.detach(), min=1e-6)
                valid_q_mask = torch.ones_like(q_scale, dtype=torch.bool)
                if tgt_key_padding_mask is not None:
                    q_scale = q_scale.masked_fill(tgt_key_padding_mask, 1.0)
                    valid_q_mask = ~tgt_key_padding_mask
                attn_bias = attn_bias + torch.log(q_scale).unsqueeze(-1)
            if memory_scale is not None:
                kv_scale = torch.clamp(memory_scale, min=1e-6)
                if memory_key_padding_mask is not None:
                    kv_scale = kv_scale.masked_fill(memory_key_padding_mask, 1.0)
                if tgt_scale is not None:
                    # Data-driven uncertainty from target trust: flatter q_scale -> less confidence.
                    q_min = q_scale.masked_fill(~valid_q_mask, float("inf")).min(dim=1, keepdim=True).values
                    q_max = q_scale.masked_fill(~valid_q_mask, float("-inf")).max(dim=1, keepdim=True).values
                    has_valid = valid_q_mask.any(dim=1, keepdim=True)
                    q_min = torch.where(has_valid, q_min, torch.ones_like(q_min))
                    q_max = torch.where(has_valid, q_max, torch.ones_like(q_max))
                    q_range = q_max - q_min
                    q_norm = (q_scale - q_min) / q_range.clamp(min=1e-6)
                    rel_uncertainty = 1.0 - q_norm
                    # For near-constant trust sequences, fallback to absolute trust level.
                    abs_uncertainty = 1.0 - torch.clamp(q_scale, min=0.0, max=1.0)
                    q_uncertainty = torch.where(q_range < 1e-6, abs_uncertainty, rel_uncertainty)
                    if tgt_key_padding_mask is not None:
                        q_uncertainty = q_uncertainty.masked_fill(tgt_key_padding_mask, 0.0)
                    kv_eff = 1.0 + q_uncertainty.unsqueeze(-1) * (kv_scale.unsqueeze(-2) - 1.0)
                    attn_bias = attn_bias + torch.log(torch.clamp(kv_eff, min=1e-6))
                else:
                    attn_bias = attn_bias + torch.log(kv_scale).unsqueeze(-2)
            attn_bias = attn_bias.repeat_interleave(self.multihead_attn.num_heads, dim=0)

        merged_memory_mask = memory_mask
        if attn_bias is not None and memory_mask is not None:
            if memory_mask.dim() == 2:
                memory_mask_exp = memory_mask.unsqueeze(0)
            else:
                memory_mask_exp = memory_mask
            if memory_mask_exp.size(0) == 1 and attn_bias.size(0) != 1:
                memory_mask_exp = memory_mask_exp.expand(attn_bias.size(0), -1, -1)
            if memory_mask_exp.dtype == torch.bool:
                mask_bias = torch.zeros_like(attn_bias)
                mask_bias = mask_bias.masked_fill(memory_mask_exp, float("-inf"))
            else:
                mask_bias = memory_mask_exp.to(device=attn_bias.device, dtype=attn_bias.dtype)
            merged_memory_mask = attn_bias + mask_bias
        elif attn_bias is not None:
            merged_memory_mask = attn_bias

        tgt2 = self.multihead_attn(tgt,
                                   memory,
                                   memory,
                                   attn_mask=merged_memory_mask,
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

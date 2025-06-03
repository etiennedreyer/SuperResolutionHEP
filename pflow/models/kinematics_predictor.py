import torch
import torch.nn as nn
import math

from models.diffusion_transformer import DiTEncoder
from models.attention import ScaledDotProductAttention
from models.dense import Dense
from models.utils import masked_softmax, merge_masks


class AttnKinematicNet(nn.Module):
    def __init__(self, config_pf):
        super().__init__()
        self.embed_dim = config_pf['h_dim']
        self.linear_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_k = nn.Linear(self.embed_dim, self.embed_dim)

        self.attention = ScaledDotProductAttention()
        self.attn_scale = math.sqrt(self.embed_dim)

    def set_trans_dicts(self, trans_dicts):
        self.trans_dicts = trans_dicts

    def forward(self, q, k, q_mask, kv_mask, attn_mask, batch):
        # Work out the masking situation, with padding, peaking, etc
        attn_mask = merge_masks(q_mask, kv_mask, attn_mask, q.shape, k.shape, q.device)

        q_proj = self.linear_q(q)
        k_proj = self.linear_k(k)

        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / self.attn_scale
        inc_weights = masked_softmax(scores, attn_mask, dim=1) # sum over particles for a cell is 1

        e_raw_inc = inc_weights * batch['cell_e_raw'].unsqueeze(1)
        
        eps_sum = (e_raw_inc.sum(dim=2, keepdim=True) == 0) # * 1e-8
        inc = e_raw_inc / (e_raw_inc.sum(dim=2, keepdim=True) + eps_sum) # sum over cells for a particle is 1

        # (b, np, nc) = (b, np, nc) * (b, 1, nc)
        eta_pred_raw = (inc * batch['cell_eta_raw'].unsqueeze(1)).sum(dim=-1)
        phi_pred = (inc * batch['cell_phi'].unsqueeze(1)).sum(dim=-1)

        e_pred_raw = e_raw_inc.sum(dim=-1)

        # zero mass assumption
        pt_pred_raw = e_pred_raw / torch.cosh(eta_pred_raw)

        # transformation
        pt  = self.trans_dicts['pt'].forward(pt_pred_raw)
        eta = self.trans_dicts['eta'].forward(eta_pred_raw)
        e   = self.trans_dicts['e'].forward(e_pred_raw)

        kin_pred = torch.cat([
            pt.unsqueeze(-1), eta.unsqueeze(-1), phi_pred.unsqueeze(-1), e.unsqueeze(-1)
        ], dim=-1)

        return kin_pred, inc_weights


class KinematicsPredictor(nn.Module):
    def __init__(self, config_pf):
        super().__init__()
        self.config_pf = config_pf
        h_dim = config_pf['h_dim']
        self.max_part = config_pf['max_particles']

        if self.config_pf['kinematics_predictor']['init_particles']['type'] == 'embedding':
            self.particle_emb_net = nn.Embedding(
                self.max_part, self.config_pf['kinematics_predictor']['init_particles']['embedding_dim']
            )
            self.particle_proj = nn.Linear(
                self.config_pf['kinematics_predictor']['init_particles']['embedding_dim'], h_dim
            )
        elif self.config_pf['kinematics_predictor']['init_particles']['type'] == 'random':
            self.edges_mu = nn.Parameter(torch.randn(1, 1, h_dim))
            self.edges_logsigma = nn.Parameter(torch.zeros(1, 1, h_dim))
            nn.init.xavier_uniform_(self.edges_logsigma)

        self.transformer = DiTEncoder(
            embed_dim=h_dim,
            num_layers=config_pf['kinematics_predictor']['transformer']['num_transformer_layers'],
            mha_config={
                "num_heads": config_pf['kinematics_predictor']['transformer']['num_heads'],
                "attention": ScaledDotProductAttention(),
            },
            dense_config=config_pf['kinematics_predictor']['transformer']['dense_config'],
            context_dim=config_pf['kinematics_predictor']['transformer']['context_size'],
        )

        self.use_attn_kin = config_pf['kinematics_predictor'].get('use_attn_kinematics', False)
        if self.use_attn_kin:
            self.kin_net = AttnKinematicNet(config_pf, )

        else:
            kin_net_config = config_pf['kinematics_predictor']['pt_eta_phi_e_net']
            self.kin_net = Dense(**kin_net_config)


    def init_particles(self, n_events, device):
        if self.config_pf['kinematics_predictor']['init_particles']['type'] == 'embedding':
            particle_index = torch.arange(
                self.max_part, device=device).unsqueeze(0).repeat(n_events, 1)
            particle_emb = self.particle_emb_net(particle_index)
            particle_emb = self.particle_proj(particle_emb)

        elif self.config_pf['kinematics_predictor']['init_particles']['type'] == 'random':
            mu = self.edges_mu.expand(n_events, self.max_part, -1)
            sigma = self.edges_logsigma.exp().expand(n_events, self.max_part, -1)
            particle_emb = mu + sigma * torch.randn(mu.shape, device=device)

        return particle_emb


    def forward(self, cell_feat, cell_mask, part_mask, batch):
        bs = cell_feat.shape[0]
        particle_emb = self.init_particles(
            bs, cell_feat.device)

        f = cell_mask.unsqueeze(-1)
        cell_global_feat = torch.sum(cell_feat * f, dim=1) / torch.sum(f, dim=1)

        part_feat = self.transformer(
            q=particle_emb, q_mask=~part_mask,
            k=cell_feat, kv_mask=~cell_mask,
            context=cell_global_feat)

        if self.use_attn_kin:
            part_kin, inc_weights = self.kin_net(
                part_feat, cell_feat, ~part_mask, ~cell_mask, attn_mask=None, batch=batch)

        else:
            part_kin = self.kin_net(part_feat)
            e_raw_inc = None

        return part_kin, inc_weights

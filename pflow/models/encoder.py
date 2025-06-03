import torch
import torch.nn as nn

from models.diffusion_transformer import DiTEncoder
from models.attention import ScaledDotProductAttention



class Encoder(nn.Module):
    def __init__(self, config_pf):
        super().__init__()
        self.config_pf = config_pf
        h_dim = config_pf['h_dim']

        self.layer_emb_net = nn.Embedding(
            3, config_pf['encoder']['layer_emb_dim']
        )
        self.cell_init_net = nn.Sequential(
            nn.Linear(8, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim)
        )

        num_transformer_layers = config_pf['encoder']['transformer']['num_transformer_layers']
        num_heads = config_pf['encoder']['transformer']['num_heads']
        self.transformer = DiTEncoder(
            embed_dim=h_dim,
            num_layers=num_transformer_layers,
            mha_config={
                "num_heads": num_heads,
                "attention": ScaledDotProductAttention(),
            },
            dense_config=config_pf['encoder']['transformer']['dense_config'],
            context_dim=config_pf['encoder']['transformer']['context_size'],
        )


    def forward(self, batch):
        cell_mask = batch['cell_mask']

        # cell initialization
        layer_emb = self.layer_emb_net(batch['cell_layer'])
        cell_feat0 = torch.cat([
            batch['cell_e'].unsqueeze(-1),
            batch['cell_eta'].unsqueeze(-1),
            batch['cell_cosphi'].unsqueeze(-1),
            batch['cell_sinphi'].unsqueeze(-1),
            layer_emb,
        ], dim=-1)
        cell_feat = self.cell_init_net(cell_feat0)
    
        f = cell_mask.unsqueeze(-1)
        cell_global_feat = torch.sum(cell_feat * f, dim=1) / torch.sum(f, dim=1)

        feat = self.transformer(
            q=cell_feat, q_mask=~cell_mask, context=cell_global_feat)

        return feat
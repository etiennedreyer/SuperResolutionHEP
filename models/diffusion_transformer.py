import torch
from torch import nn

from models.attention import MultiheadAttention
from models.dense import Dense


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



class DiTLayer(nn.Module):
    def __init__(self, embed_dim, context_dim, mha_config, dense_config=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.mha = MultiheadAttention(embed_dim, **mha_config)

        if dense_config:
            self.dense = Dense(input_size=embed_dim, output_size=embed_dim, **dense_config)
        else:
            self.register_buffer("dense", None)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(context_dim, 6 * embed_dim, bias=True))

    def forward(self, q, q_mask=None, k=None, kv_mask=None, context=None, attn_mask=None, attn_bias=None):
        '''
            if k is provided, then we will have cross-attention
        '''
        shift_msa, scale_msa, gate_msa, \
            shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(context).chunk(6, dim=1)

        if k == None: # self-attention
            q_attn = self.mha(
                q=modulate(self.norm1(q), shift_msa, scale_msa),
                q_mask=q_mask, attn_mask=attn_mask, attn_bias=attn_bias)
        
        else: # cross-attention
            q_attn = self.mha(
                q=q, k=modulate(self.norm1(k), shift_msa, scale_msa),
                q_mask=q_mask, kv_mask=kv_mask, attn_mask=attn_mask, attn_bias=attn_bias)
            
        q = q + gate_msa.unsqueeze(1) * q_attn
        
        if self.dense:
            q_mlp = self.dense(modulate(self.norm2(q), shift_mlp, scale_mlp), context)
            q = q + gate_mlp.unsqueeze(1) * q_mlp

        return q



class DiTEncoder(nn.Module):
    def __init__(
        self, embed_dim, num_layers, mha_config,
        dense_config=None, context_dim=0, out_dim=0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.out_dim = out_dim

        self.layers = nn.ModuleList(
            [DiTLayer(
                embed_dim, context_dim, mha_config, dense_config,
            ) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        # For resizing the output tokens
        if self.out_dim:
            self.final_linear = nn.Linear(self.embed_dim, self.out_dim)


    def forward(self, q, **kwargs):
        for layer in self.layers:
            q = layer(q, **kwargs)
        q = self.final_norm(q)

        # Optinal resizing layer
        if self.out_dim:
            q = self.final_linear(q)
        return q

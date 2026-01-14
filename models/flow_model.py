import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import TransformerEncoder
from models.diffusion_transformer import DiTEncoder
from models.attention import ScaledDotProductAttention
from models.dense import Dense
from models.utils import TimestepEmbedder

import torchdiffeq
from torchcfm.conditional_flow_matching import TargetConditionalFlowMatcher

import numpy as np



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



class FlowModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        
        self.config = model_config

        ### Experimental: testing how x1 regression compares to flow matching
        self.one_shot = self.config.get('one_shot', False)
        if self.one_shot:
            print('\033[96m' + 'Using one-shot energy regression model' + '\033[0m')

        self.n_steps = self.config['n_steps']

        self.sigma_min = self.config['sigma_min']
        self.flow_match_obj = TargetConditionalFlowMatcher(sigma=self.sigma_min)

        if not self.one_shot:
            self.time_step_embedder = TimestepEmbedder(
                self.config['time_embedding_size'])

        self.h_dim = int(self.config['h_dim'])
        self.context_size = self.config['time_embedding_size'] + \
            self.config['etaphi_emb']['output_size'] + \
            self.config['layer_emb']['dense_config']['output_size'] + \
            self.config['e_proxy_emb']['output_size'] + 1
        if self.one_shot:
            self.context_size = 0
        else:
            self.context_size = self.config['time_embedding_size']

        etaphi_emb_config = self.config['etaphi_emb']
        etaphi_emb_config['context_size'] = self.context_size ##
        self.etaphi_emb_net = Dense(**etaphi_emb_config)

        layer_emb_config = self.config['layer_emb']
        layer_emb_config['dense_config']['context_size'] = self.context_size ##
        self.layer_emb_table = nn.Embedding(3, layer_emb_config['emb_dim'])
        self.layer_emb_net = Dense(**layer_emb_config['dense_config'])

        e_proxy_emb_config = self.config['e_proxy_emb']
        e_proxy_emb_config['context_size'] = self.context_size ##
        self.proxy_emb_net = Dense(**e_proxy_emb_config)

        self.cond_emb_dim = etaphi_emb_config['output_size'] + \
            layer_emb_config['dense_config']['output_size'] + \
            e_proxy_emb_config['output_size'] + 1

        noisy_input_emb_config = self.config['noisy_input_emb']
        noisy_input_emb_config['context_size'] = self.context_size
        self.noisy_input_emb_net = Dense(**noisy_input_emb_config)

        self.context_size_plus = self.context_size + self.cond_emb_dim

        if self.one_shot:
            noisy_input_emb_config['output_size'] = 0

        # needed purely for dimensionality matching
        feat_0_mlp_config = self.config['feat_0_mlp']
        if feat_0_mlp_config['input_size'] == -1:
            feat_0_mlp_config['input_size'] = \
                etaphi_emb_config['output_size'] + layer_emb_config['dense_config']['output_size'] + \
                e_proxy_emb_config['output_size'] + noisy_input_emb_config['output_size'] + 1
            # feat_0_mlp_config['input_size'] = 10
        feat_0_mlp_config['context_size'] = self.context_size_plus
        self.feat_0_mlp = Dense(**feat_0_mlp_config)

        if self.config['transformer']['type'] == 'GPT-2+Normformer':
            self.transformer = TransformerEncoder(
                embed_dim =self.config['h_dim'],
                num_layers=self.config['transformer']['num_transformer_layers'],
                mha_config={
                    'num_heads': self.config['transformer']['num_heads'],
                    'attention': ScaledDotProductAttention()
                },
                dense_config=self.config['transformer']['dense_config'],
                context_dim=self.context_size_plus
            )

        elif self.config['transformer']['type'] == 'DiT':
            self.transformer = DiTEncoder(
                embed_dim=self.h_dim,
                num_layers=self.config['transformer']['num_transformer_layers'],
                mha_config={
                    'num_heads': self.config['transformer']['num_heads'],
                    'attention': ScaledDotProductAttention()
                },
                dense_config=self.config['transformer']['dense_config'],
                context_dim=self.context_size_plus
            )

        self.v_t_input_dim = self.h_dim + self.cond_emb_dim
        if self.config.get('final_modulation', False):
            self.norm_v_t = nn.LayerNorm(self.v_t_input_dim)
            self.v_t_adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(self.context_size_plus, 2 * self.v_t_input_dim, bias=True))

        v_t_pred_config = self.config['v_t_pred']
        v_t_pred_config['input_size'] = self.v_t_input_dim
        v_t_pred_config['context_size'] = self.context_size_plus
        self.v_t_pred_net = Dense(**v_t_pred_config)

        self.initialize_weights()
        # self.get_param_summary()


    def initialize_weights(self):
        # initialize linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        if self.config['init_weights'].get('all_linear', None) == 'xavier_uniform':
            print('\033[96m' + 'Initializing all linear weights with xavier_uniform' + '\033[0m')
            self.apply(_basic_init)


        # Initialize layer embedding table:
        if self.config['init_weights'].get('layer_emb_table', None) == 'normal':
            print('\033[96m' + 'Initializing layer embedding table with normal (std=0.02)' + '\033[0m')
            nn.init.normal_(self.layer_emb_table.weight, std=0.02)

        if not self.one_shot:
            # Initialize timestep embedding MLP:
            if self.config['init_weights'].get('time_step_embedder', None) == 'normal':
                print('\033[96m' + 'Initializing time_step_embedder with normal (std=0.02)' + '\033[0m')
                nn.init.normal_(self.time_step_embedder.mlp[0].weight, std=0.02)
                nn.init.normal_(self.time_step_embedder.mlp[2].weight, std=0.02)

        # zero-out the adaLN modulation layers in DiTLayers
        if self.config['init_weights'].get('ln_modulation', None) == 'zero':
            print('\033[96m' + 'Zeroing out adaLN modulation layers' + '\033[0m')
            for layer in self.transformer.layers:
                nn.init.constant_(layer.adaLN_modulation[1].weight, 0)
                nn.init.constant_(layer.adaLN_modulation[1].bias, 0)

            # zero-out the v_t layers
            nn.init.constant_(self.v_t_adaLN_modulation[1].weight, 0)
            nn.init.constant_(self.v_t_adaLN_modulation[1].bias, 0)

        # zero-out the final linear layer in v_t_pred_net
        if self.config['init_weights'].get('v_t_pred_linear', None) == 'zero':
            print('\033[96m' + 'Zeroing out v_t_pred_net final linear layer' + '\033[0m')
            nn.init.constant_(self.v_t_pred_net.net[-1].weight, 0)
            nn.init.constant_(self.v_t_pred_net.net[-1].bias, 0)


    def get_stat(self, tensor):
        ret_dict = {
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'mean': tensor.mean().item(),
            'std': tensor.std().item()
        }
        return ret_dict


    def forward(self, batch, noisy_input, time_step, verbose=False):
        '''
            noisy_input: can work with both approaches -
                - noising the target energy
                - noising the (target-proxy) energy
        '''
        if self.one_shot:
            ### won't be using these guys
            noisy_input = None
            time_emb = None
        else:
            # time embedding
            time_emb = self.time_step_embedder(time_step)

        if verbose:
            torch.set_printoptions(precision=3)
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    print(key, val.shape, torch.isfinite(val).all())

        if verbose and not self.one_shot:
            print('time_emb:')
            print('\t', self.get_stat(time_emb))
        
        # parse batch
        eta, cosphi, sinphi, layer = batch['eta'], batch['cosphi'], batch['sinphi'],  batch['layer']
        e_proxy = batch['e_proxy']
        q_mask, edge_mask = batch['q_mask'], batch['edge_mask']

        # encode conditional info
        layer_emb = self.layer_emb_table(layer.squeeze(-1))
        layer_emb = self.layer_emb_net(layer_emb, context=time_emb)
        etaphi_emb  = self.etaphi_emb_net(torch.cat([eta, cosphi, sinphi], dim=2), context=time_emb)
        e_proxy_emb = self.proxy_emb_net(e_proxy, context=time_emb)

        if verbose:
            print('layer_emb:')
            print('\t', self.get_stat(layer_emb))

            print('etaphi_emb:')
            print('\t', self.get_stat(etaphi_emb))

            print('e_proxy_emb:')
            print('\t', self.get_stat(e_proxy_emb))

        cond_feat = torch.cat([
            etaphi_emb, layer_emb, e_proxy_emb, e_proxy
        ], dim=-1)
        cond_feat_global = torch.sum(cond_feat * q_mask.unsqueeze(-1), dim=1) / \
            torch.sum(q_mask, dim=1, keepdim=True)


        if self.one_shot:
            noisy_input_emb = None
        else:
            # encode noisy input
            noisy_input_emb = self.noisy_input_emb_net(noisy_input, context=time_emb)

        if verbose and not self.one_shot:
            print('noisy_input_emb:')
            print('\t', self.get_stat(noisy_input_emb))

        if self.one_shot:
            context = cond_feat_global
            feat_0 = cond_feat
        else:
            # context (t, global)
            context = torch.cat([time_emb, cond_feat_global], dim=-1)

            feat_0 = torch.cat([
                cond_feat, noisy_input_emb
            ], dim=-1)

        feat = self.feat_0_mlp(feat_0, context=context)

        if verbose:
            print('feat (after feat_0_mlp):')
            print('\t', self.get_stat(feat))

        feat = self.transformer(q=feat, q_mask=~q_mask, attn_mask=None, context=context)

        if verbose:
            print('feat (after transformer):')
            print('\t', self.get_stat(feat))

        # final skip connection
        feat = torch.cat([feat, cond_feat], dim=-1)
        
        if self.config.get('final_modulation', False):
            v_t_shift, v_t_scale = self.v_t_adaLN_modulation(context).chunk(2, dim=1)
            feat = modulate(self.norm_v_t(feat), v_t_shift, v_t_scale)

            if verbose:
                print('v_t_shift:')
                print('\t', self.get_stat(v_t_shift))

                print('v_t_scale:')
                print('\t', self.get_stat(v_t_scale))

                print('feat (after modulation):')
                print('\t', self.get_stat(feat))


        v_t = self.v_t_pred_net(feat, context=context)

        if verbose:
            print('v_t:')
            print('\t', self.get_stat(v_t))

        return v_t


    def get_loss(self, batch):
        '''
            t=0: noise data, t=1: real data
        '''
        target = batch['target']
        if self.one_shot:
            xt = None
            t = None
            ut = target
        else:
            x_0 = torch.randn_like(target)
            # t = torch.Tensor(np.random.power(2, size=x_0.shape[0])).to(x_0.device)
            t = torch.rand((x_0.size(0),), dtype=x_0.dtype, device=x_0.device)
            t, xt, ut = self.flow_match_obj.sample_location_and_conditional_flow(x_0, target, t=None)

        vt = self.forward(batch, xt, t)
        loss = F.mse_loss(vt, ut, reduction='none')[batch['q_mask']]
        # loss = 2 * F.huber_loss(vt, ut, reduction='none')[batch['q_mask']]

        loss_detached = loss.clone().detach()

        if not torch.isfinite(loss).all():
            self.forward(batch, xt, t, verbose=True)
            exit(0)


        ret_dict = {
            'ut_max': ut.max().item(), 'ut_min': ut.min().item(),
            'ut_mean': ut.mean().item(), 'ut_std': ut.std().item(),
            'vt_max': vt.max().item(), 'vt_min': vt.min().item(),
            'vt_mean': vt.mean().item(), 'vt_std': vt.std().item(),
            'loss_max': loss.max().item(), 'loss_min': loss.min().item(),
            'loss_mean': loss.mean().item(), 'loss_std': loss.std().item()
        }

        loss = loss.mean()

        return loss, ret_dict, loss_detached


    @torch.no_grad()
    def generate_samples(self, batch, n_steps=None, method="dopri5", ret_seq=False):
        '''
            batch: batch of data
            n_steps: number of steps to integrate
            method: integration method
            ret_seq: return the entire sequence or just the last step
        '''
        if self.one_shot:
            return self.forward(batch, None, None)

        if n_steps is None:
            n_steps = self.n_steps

        proxy_e = batch['e_proxy']

        x_1 = torchdiffeq.odeint(
            lambda t, x_t: self.forward(
                batch, x_t, time_step = t * torch.ones(x_t.shape[0], device=proxy_e.device)
            ),
            torch.randn_like(proxy_e, device=proxy_e.device), # x_0
            torch.linspace(0, 1, n_steps, device=proxy_e.device), # t
            method=method,
            atol=1e-4,
            rtol=1e-4,
        )

        if not ret_seq:
            x_1 = x_1[-1, ...]

        return x_1
    

    def get_param_summary(self):
        ret_dict = {}
        for name, module in self._modules.items():
            weights = []
            biases = []
            for sub_module in module.modules():
                if isinstance(sub_module, nn.Linear):
                    weights.append(sub_module.weight.view(-1))
                    if sub_module.bias is not None:
                        biases.append(sub_module.bias.view(-1))
            if weights:
                weights = torch.cat(weights)
                tmp_dict = {
                    'weight': {
                        'min': weights.min().item(),
                        'max': weights.max().item(),
                        'mean': weights.mean().item(),
                        'std': weights.std().item()
                    }
                }

                if biases:
                    biases = torch.cat(biases)
                    tmp_dict['bias'] = {
                        'min': biases.min().item(),
                        'max': biases.max().item(),
                        'mean': biases.mean().item(),
                        'std': biases.std().item()
                    }
                ret_dict[name] = tmp_dict
            
        return ret_dict

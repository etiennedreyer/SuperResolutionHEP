import torch
import torch.nn as nn

from .encoder import Encoder
from .cardinality_predictor import CardinalityPredictor
from .kinematics_predictor import KinematicsPredictor


class SAPF(nn.Module):
    def __init__(self, config_pf, inference=False):
        super().__init__()
        self.config_pf = config_pf
        self.encoder = Encoder(config_pf)
    
        if config_pf.get('cardinality_predictor', None) is not None:
            self.cardinality_predictor = CardinalityPredictor(config_pf)
    
        if config_pf.get('kinematics_predictor', None) is not None:
            self.kinematics_predictor = KinematicsPredictor(config_pf)
        
        self.initialize_weights()
        self.inference = inference


    def initialize_weights(self):
        # initialize linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        if self.config_pf['init_weights'].get('all_linear', None) == 'xavier_uniform':
            print('\033[96m' + 'Initializing all linear weights with xavier_uniform' + '\033[0m')
            self.apply(_basic_init)


        # Initialize layer embedding table:
        if self.config_pf['init_weights'].get('layer_emb_table', None) == 'normal':
            print('\033[96m' + 'Initializing layer embedding table with normal (std=0.02)' + '\033[0m')
            nn.init.normal_(self.encoder.layer_emb_net.weight, std=0.02)


        # zero-out the adaLN modulation layers in DiTLayers
        if self.config_pf['init_weights'].get('ln_modulation', None) == 'zero':
            print('\033[96m' + 'Zeroing out adaLN modulation layers' + '\033[0m')
            for layer in self.encoder.transformer.layers:
                nn.init.constant_(layer.adaLN_modulation[1].weight, 0)
                nn.init.constant_(layer.adaLN_modulation[1].bias, 0)

            if self.config_pf.get('kinematics_predictor', None) is not None:
                for layer in self.kinematics_predictor.transformer.layers:
                    nn.init.constant_(layer.adaLN_modulation[1].weight, 0)
                    nn.init.constant_(layer.adaLN_modulation[1].bias, 0)


    def forward(self, batch):
        encoded_feat = self.encoder(batch)
    
        n_pred_logits = None
        if self.config_pf.get('cardinality_predictor', None) is not None:    
            n_pred_logits = self.cardinality_predictor(encoded_feat, batch['cell_mask'])

        kin_pred = None; inc_weights = None
        if self.config_pf.get('kinematics_predictor', None) is not None:
            if self.inference:
                n_pred = torch.argmax(n_pred_logits, dim=-1)
                part_mask = torch.arange(self.config_pf['max_particles'], device=n_pred.device).unsqueeze(0) < n_pred.unsqueeze(1)
            else:
                part_mask = batch['part_mask']

            kin_pred, inc_weights = self.kinematics_predictor(
                encoded_feat, batch['cell_mask'], part_mask, batch)

        return n_pred_logits, kin_pred, inc_weights

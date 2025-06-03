import torch
import torch.nn as nn
from models.dense import Dense


class CardinalityPredictor(nn.Module):
    def __init__(self, config_pf):
        super().__init__()
        self.config_pf = config_pf
        self.max_part = config_pf['max_particles'] + 1 # 0 class means no particle

        card_pred_config = self.config_pf['cardinality_predictor']
        card_pred_config['output_size'] = self.max_part
        self.card_pred_net = Dense(**card_pred_config)


    def forward(self, embedded_feat, cell_mask):
        f = cell_mask.unsqueeze(-1)
        global_feat = torch.sum(embedded_feat * f, dim=1) / torch.sum(f, dim=1)
        n_pred_logits = self.card_pred_net(global_feat)

        return n_pred_logits

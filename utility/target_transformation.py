import torch
from utility.transformation import VarTransformation

class TargetTransformation(VarTransformation):
    def __init__(self, config):
        super().__init__(config)

    def trans(self, hr_truth_raw, proxy_raw=None, **kwargs):
        if self.transformation == 'logit_ratio':    
            assert proxy_raw is not None, 'proxy_raw must be provided'
            ratio = hr_truth_raw / (proxy_raw * self.f)
            ratio = torch.clamp(ratio, 0, 1.0)
            ratio = self.alpha + (1 - 2*self.alpha) * ratio
            logit = torch.log((ratio) / (1 - ratio))
            return logit

    def inv_trans(self, nn_out, proxy_raw=None, **kwargs):
        if self.transformation == 'logit_ratio':
            assert proxy_raw is not None, 'proxy_raw must be provided'
            ratio = 1 / (1 + torch.exp(-nn_out))
            ratio = (ratio - self.alpha) / (1 - 2*self.alpha)
            e_pred_raw = ratio * proxy_raw * self.f
            return e_pred_raw

    def forward(self, hr_truth_raw, proxy_raw=None, **kwargs):
        x = self.trans(hr_truth_raw, proxy_raw, **kwargs)
        x = self.scale(x)
        return x
    
    def inverse(self, nn_out, proxy_raw=None, **kwargs):
        x = self.inv_scale(nn_out)
        x = self.inv_trans(x, proxy_raw, **kwargs)
        return x

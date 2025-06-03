import torch

class VarTransformation:
    '''
        trans: tranforming the quantities
            eg. x -> log(x), pow(e,m) etc
        scale: scaling the quantities
            eg. x -> (x - mean(x)) / std(x)
        forward: trans + scale
    '''

    def __init__(self, config):
        self.config = config
        for k, v in config.items():
            if isinstance(v, torch.Tensor): # random unexplainable bug otherwise
                v = v.item()
            setattr(self, k, v)

    def trans(self, x):
        if self.transformation == None:
            return x
        elif self.transformation == 'pow(x,m)':
            return torch.pow(x, self.m)

        elif self.transformation == 'pow(x,m)_signed':
            sign = (x >= 0) * 2 - 1 # agnstic to torch/np
            return sign * (abs(x) ** self.m)

    def inv_trans(self, x):
        if self.transformation == None:
            return x
        elif self.transformation == 'pow(x,m)':
            return torch.pow(x, 1 / self.m)

        elif self.transformation == 'pow(x,m)_signed':
            sign = (x >= 0) * 2 - 1
            return sign * (abs(x) ** (1 / self.m))


    def scale(self, x):
        if self.scale_mode == None:
            return x
        elif self.scale_mode == 'min_max':
            x = (x - self.min) / (self.max - self.min) # [0,1]
            target_min, target_max = self.range
            return x * (target_max - target_min) + target_min
        elif self.scale_mode == 'standard':
            return (x - self.mean) / self.std
    
    def inv_scale(self, x):
        if self.scale_mode == None:
            return x
        elif self.scale_mode == 'min_max':
            target_min, target_max = self.range
            x = (x - target_min) / (target_max - target_min) # [0,1]
            return x * (self.max - self.min) + self.min
        elif self.scale_mode == 'standard':
            return x * self.std + self.mean


    def forward(self, x):
        x = self.trans(x)
        x = self.scale(x)
        return x
    
    def inverse(self, x):
        x = self.inv_scale(x)
        x = self.inv_trans(x)
        return x

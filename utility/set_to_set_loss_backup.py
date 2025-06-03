import numpy as np
import ray
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import os



def l_split_ind(l, n):
    r = l%n
    return np.cumsum([0] + [l//n+1]*r + [l//n]*(n-r))

@ray.remote
def lsa(arr, s, e):
    return np.array([linear_sum_assignment(p) for p in arr[s:e]])

def ray_lsa(arr, n):
    l = arr.shape[0]
    ind = l_split_ind(l, n)
    arr_id = ray.put(arr)
    res = [lsa.remote(arr_id, ind[i], ind[i+1]) for i in range(n)]
    res = np.concatenate([ray.get(r) for r in res])
    return res



class SetToSetLoss:

    def __init__(self, config, max_part):
        self.EPS = 1e-8

        self.max_part    = max_part
        self.E_LOSS_WT   = config['e_loss_wt']
        self.PT_LOSS_WT  = config['pt_loss_wt']
        self.ETA_LOSS_WT = config['eta_loss_wt']
        self.PHI_LOSS_WT = config['phi_loss_wt']


    def get_loss_mask(self, part_mask_float):
        '''
            part_mask.shape = (bs, max_part)

            loss should look like (for 3 real particles and 2 padded)
                
                    l    l    l    inf inf
                    l    l    l    inf inf
                L = l    l    l    inf inf
                    inf  inf  inf  0   0
                    inf  inf  inf  0   0

            adding part_mask will give us -

                         2 (RR)    2 (RR)    2 (RR)    1 (RF)    1 (RF)
                         2 (RR)    2 (RR)    2 (RR)    1 (RF)    1 (RF)
                _mask =  2 (RR)    2 (RR)    2 (RR)    1 (RF)    1 (RF)
                         1 (RF)    1 (RF)    1 (RF)    0 (FF)    0 (FF)
                         1 (RF)    1 (RF)    1 (RF)    0 (FF)    0 (FF)

                _mask = q1 q2
                        q3 q4

            L = L * (~q4)   +   (q2_q3_mask * inf)
              = L * not_q4  +   q2_q3_inf

        '''
        _sum_mask = part_mask_float.unsqueeze(1).expand(-1, self.max_part, -1) + \
            part_mask_float.unsqueeze(2).expand(-1, -1, self.max_part)

        q2_q3_inf = (_sum_mask == 1) * 1e6
        not_q4 = (_sum_mask != 0).float()

        return not_q4, q2_q3_inf



    def compute(self, input, batch, n_ray=0):
        
        bs = input.size(0)

        # (b, n, 3) -> (b, n) -> (b, 1, n) -> (b, n, m)
        input_pt_reshaped  = input[:, :, 0].unsqueeze(1).expand(-1, self.max_part, -1)
        input_eta_reshaped = input[:, :, 1].unsqueeze(1).expand(-1, self.max_part, -1)
        input_phi_reshaped = input[:, :, 2].unsqueeze(1).expand(-1, self.max_part, -1)
        input_e_reshaped   = input[:, :, 3].unsqueeze(1).expand(-1, self.max_part, -1)

        target_pt_reshaped  = batch['part_pt'].unsqueeze(2).expand(-1, -1, self.max_part)
        target_eta_reshaped = batch['part_eta'].unsqueeze(2).expand(-1, -1, self.max_part)
        target_phi_reshaped = batch['part_phi'].unsqueeze(2).expand(-1, -1, self.max_part)
        target_e_reshaped   = batch['part_dep_e'].unsqueeze(2).expand(-1, -1, self.max_part) # dep_e not e

        pt_loss  = self.PT_LOSS_WT  * F.mse_loss(input_pt_reshaped, target_pt_reshaped, reduction='none')
        eta_loss = self.ETA_LOSS_WT * F.mse_loss(input_eta_reshaped, target_eta_reshaped, reduction='none')
        phi_loss = self.PHI_LOSS_WT * (1 - torch.cos(input_phi_reshaped - target_phi_reshaped))
        e_loss   = self.E_LOSS_WT   * F.mse_loss(input_e_reshaped, target_e_reshaped, reduction='none')

        not_q4, q2_q3_inf = self.get_loss_mask(batch['part_mask'].float())
        pt_loss  = pt_loss * not_q4 + q2_q3_inf
        eta_loss = eta_loss * not_q4 + q2_q3_inf
        phi_loss = phi_loss * not_q4 + q2_q3_inf
        e_loss   = e_loss * not_q4 + q2_q3_inf

        pdist = e_loss + pt_loss + eta_loss + phi_loss
        pdist_ = pdist.detach().cpu().numpy()

        if n_ray > 0:
            indices = ray_lsa(pdist_, n_ray)
        else:
            indices = np.array([linear_sum_assignment(p) for p in pdist_])

        assignment_indices = indices # shape (bs*t_bptt, 2, n_particles)

        indices = indices.shape[2] * indices[:, 0] + indices[:, 1]
        losses = torch.gather(pdist.flatten(1,2), 1, torch.from_numpy(indices).to(device=pdist.device))
        total_loss = losses.mean(1).mean(0)

        # for book-keeping
        e_losses = torch.gather(e_loss.flatten(1,2), 1, torch.from_numpy(indices).to(device=pdist.device))
        e_loss = e_losses.mean(1).mean(0)

        pt_losses = torch.gather(pt_loss.flatten(1,2), 1, torch.from_numpy(indices).to(device=pdist.device))
        pt_loss = pt_losses.mean(1).mean(0)

        eta_losses = torch.gather(eta_loss.flatten(1,2), 1, torch.from_numpy(indices).to(device=pdist.device))
        eta_loss = eta_losses.mean(1).mean(0)

        phi_losses = torch.gather(phi_loss.flatten(1,2), 1, torch.from_numpy(indices).to(device=pdist.device))
        phi_loss = phi_losses.mean(1).mean(0)

        loss_componenets = {
            'e_loss': e_loss.item(),
            'pt_loss': pt_loss.item(),
            'eta_loss': eta_loss.item(),
            'phi_loss': phi_loss.item()
        }

        assgn_indices = assignment_indices[:, 1, :]

        return total_loss, loss_componenets, assgn_indices
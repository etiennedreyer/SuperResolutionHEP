
# import comet_ml

from pytorch_lightning.core.module import LightningModule

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.flow_model import FlowModel

from utility.live_plotting_util import graph2img_scd, PerformanceCOCOALive
from utility.custom_lr_scheduler import CustomLRScheduler
from utility.transformation import VarTransformation
from utility.target_transformation import TargetTransformation
from utility.sampler import SuperResSampler

from dataset import SupResDataset, collate_graphs, collate_graphs_plus

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from pytorch_lightning.utilities import grad_norm, rank_zero_only



class SupResLightning(LightningModule):

    def __init__(self, config_mv, config_t, comet_logger=None):
        super().__init__()

        self.save_hyperparameters()

        self.config_mv = config_mv
        self.config_t = config_t
        self.net = FlowModel(self.config_mv['flow_model'])

        self.comet_logger = comet_logger        
        self.automatic_optimization = True # False
        self.batch_size_cum = 0

        self.transform_dicts = {}
        for k, v in self.config_mv['var_transform'].items():
            self.transform_dicts[k] = VarTransformation(v)
        self.target_trans_obj = TargetTransformation(self.config_mv['target_transform'])

        self.perf_live = PerformanceCOCOALive(
            self.config_mv['res_factor'], self.transform_dicts, self.target_trans_obj)

        self.stored_loss = None
        self.stored_idxs = None

        self._val_outputs = []

    def set_comet_logger(self, comet_logger):
        self.comet_logger = comet_logger


    def train_dataloader(self):
        reduce_ds = self.config_t['reduce_ds_train']
        ds = SupResDataset(self.config_t['train_path'], reduce_ds=reduce_ds, 
            config_mv=self.config_mv, one_event_train=self.config_t['one_event_train'],
            one_event_idx=self.config_t['one_event_idx'])
        
        if self.config_t.get('use_sampler', False):    
            batch_sampler = SuperResSampler(np.array(ds.cell_count_high),
                batch_size=self.config_t['batch_size_train'], 
                n_sq_sum_threshold=self.config_t['n_sq_sum_threshold_train'], drop_last=False)
            loader = DataLoader(ds, num_workers=self.config_t["num_workers"],
                collate_fn=collate_graphs, batch_sampler=batch_sampler, pin_memory=True, persistent_workers=True)
        else:
            loader = DataLoader(ds, batch_size=self.config_t['batch_size_train'], 
                num_workers=self.config_t['num_workers'], shuffle=True, collate_fn=collate_graphs, persistent_workers=True)

        return loader

    
    def val_dataloader(self):
        reduce_ds = self.config_t['reduce_ds_val']
        ds = SupResDataset(self.config_t['val_path'], reduce_ds=reduce_ds, 
            config_mv=self.config_mv, one_event_train=self.config_t['one_event_train'],
            one_event_idx=self.config_t['one_event_idx'], make_low_graph=True)
        
        if self.config_t.get('use_sampler', False):
            batch_sampler = SuperResSampler(np.array(ds.cell_count_high),
                batch_size=self.config_t['batch_size_val'], 
                n_sq_sum_threshold=self.config_t['n_sq_sum_threshold_val'], drop_last=False)
            loader = DataLoader(ds, num_workers=self.config_t["num_workers"],
                collate_fn=collate_graphs_plus, batch_sampler=batch_sampler, pin_memory=True, persistent_workers=True)
        else:
            loader = DataLoader(ds, batch_size=self.config_t['batch_size_val'], 
                num_workers=self.config_t['num_workers'], shuffle=False, collate_fn=collate_graphs_plus, persistent_workers=True)

        return loader


    def training_step(self, batch, batch_idx):
        with torch.autograd.set_detect_anomaly(True):
            loss, _dict , loss_detached = self.net.get_loss(batch)
            self.log_dict(_dict, on_step=False, on_epoch=True)

            self.stored_loss = loss_detached
            self.stored_idxs = batch['idx']

            bs = batch['q_mask'].shape[0]
            self.log('train/loss', loss.item(), batch_size=bs, on_step=False, on_epoch=True)
            return loss.mean()


    def validation_step(self, batch, batch_idx):
        q_mask, target = batch['q_mask'], batch['target']

        return_dict = {}
        with torch.no_grad():
            pred = self.net.generate_samples(batch)

            loss = F.mse_loss(
                target[q_mask][:, 0], pred[q_mask][:, 0])
            return_dict['val_loss'] = loss.item()

            loss_raw_sum = 0
            for bs_i in range(batch['q_mask'].shape[0]):
                q_mask_bs_i = q_mask[bs_i]

                e_truth_raw = batch['e_truth_raw'][bs_i][q_mask_bs_i]
                pred_bs_i = pred[bs_i][q_mask_bs_i]
                
                e_pred_raw = self.target_trans_obj.inverse(
                    pred_bs_i, batch['e_proxy_raw'][bs_i][q_mask_bs_i])

                loss_raw_sum += F.mse_loss(e_truth_raw, e_pred_raw, reduction='sum').item()

            return_dict['val_loss_raw'] = loss_raw_sum / q_mask.sum().item()            
            return_dict['val_loss_n_nodes'] = q_mask.sum().item()

            if batch_idx == 0:
                return_dict['plot_data_dicts'] = []
                for bs_i in range(batch['q_mask'].shape[0]):
                    if bs_i == self.config_t.get('n_event_displays', -1):
                        break

                    q_mask_bs_i = q_mask[bs_i]

                    tmp_dict = {}
                    for k in ['eta_raw', 'phi', 'layer', 'target', 'e_truth_raw']:
                        tmp_dict[k] = batch[k][bs_i][q_mask_bs_i]
                    tmp_dict['pred'] = pred[bs_i][q_mask_bs_i]

                    pred_bs_i = pred[bs_i][q_mask_bs_i]

                    e_pred_raw = self.target_trans_obj.inverse(
                        pred_bs_i, batch['e_proxy_raw'][bs_i][q_mask_bs_i])                    
                    tmp_dict['e_pred_raw'] = e_pred_raw

                    for k, v in tmp_dict.items():
                        tmp_dict[k] = v.detach().cpu().numpy()
                        
                    return_dict['plot_data_dicts'].append(tmp_dict)

            self.perf_live.update(batch, pred)

        self._val_outputs.append(return_dict)

        return return_dict


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config_t['learningrate'])

        if self.config_t['lr_scheduler'] == None:
            optimizer.zero_grad()
            return {"optimizer": optimizer}

        warm_start_epochs = self.config_t['lr_scheduler']['warm_start_epochs']
        cosine_epochs = self.config_t['lr_scheduler']['cosine_epochs']
        eta_min = self.config_t['lr_scheduler']['eta_min']
        last_epoch = self.config_t['lr_scheduler']['last_epoch']
        max_epoch = None
        if self.config_t['lr_scheduler']['max_epochs'] == 'take_as_num_epochs':
            max_epoch = self.config_t['num_epochs']
        scheduler = CustomLRScheduler(optimizer, warm_start_epochs, cosine_epochs, eta_min, last_epoch, max_epoch)

        # we don't do it in the beginning of training_step because we want to accumulate the gradients
        optimizer.zero_grad()

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    # def on_before_optimizer_step(self, optimizer, optimizer_idx):
    #     # Compute the 2-norm for each layer
    #     norms = grad_norm(self.net, norm_type=2)
    #     norms2store = {
    #         'grad_2.0_norm_total': norms['grad_2.0_norm_total']
    #     }
    #     # find the maximum norm
    #     max_norm = 0; max_norm_key = ''
    #     for k, v in norms.items():
    #         if 'grad_2.0_norm' in k and 'total' not in k:
    #             if v > max_norm:
    #                 max_norm = v
    #                 max_norm_key = k

    #     norms2store['grad_2.0_norm_max'] = max_norm
    #     if 'grad_2.0_norm' in max_norm_key:
    #         norms2store[max_norm_key] = norms[max_norm_key]

    #     if max_norm == 0:
    #         print('max norm grad is zero')
    #         for k, v in norms.items():
    #             print('\t', k, v)
            
    #         print()
    #         print('stored_loss')
    #         print(f'\tmin: {self.stored_loss.min().item()}')
    #         print(f'\tmax: {self.stored_loss.max().item()}')
    #         print(f'\tmean: {self.stored_loss.mean().item()}')
    #         print(f'\tstd: {self.stored_loss.std().item()}')
    #         print(f'\tshape: {self.stored_loss.shape}')
    #         print(f'\tfinite_count: {torch.isfinite(self.stored_loss).sum().item()}')

    #         print('stored_idxs')
    #         print(f'\t{self.stored_idxs}')

    #     self.log_dict(norms2store)


    def on_train_epoch_end(self):
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        if self.config_t['lr_scheduler'] is not None:
            self.lr_schedulers().step()


    def on_validation_epoch_end(self):

        if self.trainer.sanity_checking:
            print("exiting")
            return
        outputs = self._val_outputs

        _val_loss_n_nodes = np.array([out['val_loss_n_nodes'] for out in outputs])
        _val_loss = np.array([out['val_loss'] for out in outputs])
        val_loss = (_val_loss * _val_loss_n_nodes).sum() / _val_loss_n_nodes.sum()
        self.log('val/loss', val_loss)

        _val_loss_raw = np.array([out['val_loss_raw'] for out in outputs])
        val_loss_raw = (_val_loss_raw * _val_loss_n_nodes).sum() / _val_loss_n_nodes.sum()
        self.log('val/loss_raw', val_loss_raw)

        # event displays
        for p_i, pl_dict in enumerate(outputs[0]['plot_data_dicts']):
            if p_i == self.config_t.get('n_event_displays', -1):
                break
            fig = plt.figure(figsize=(16.5, 7.5), dpi=100, tight_layout=True)
            graph2img_scd(pl_dict, fig)
            self.log_image(fig, f'ED_{p_i}')

        # summary plot
        fig, ev_summ_dict = self.perf_live.plot_residual_event()
        self.log_image(fig, 'residual_event_energy')
        self.log_dict(ev_summ_dict)

        fig = self.perf_live.plot_residual_cell()
        self.log_image(fig, 'residual_cell_energy')

        self.perf_live.reset()
        self._val_outputs.clear()
            

    @rank_zero_only
    def log_image(self, fig, name):
        if self.comet_logger is not None:
            canvas = FigureCanvas(fig)
            canvas.draw()
            w, h = fig.get_size_inches() * fig.get_dpi()
            image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(h), int(w), 3)
            self.comet_logger.experiment.log_image(
                image_data=image,
                name=name,
                overwrite=False, 
                image_format="png",
            )
        else:
            plt.savefig(f'plot_dump/{name}.png')
        plt.close(fig)
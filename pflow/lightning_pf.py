
from pytorch_lightning.core.module import LightningModule

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .dataset_pf import PflowDataset, collate_fn
from .models.model_pf import SAPF
from utility.custom_lr_scheduler import CustomLRScheduler
from utility.sampler import SuperResSampler
from utility.set_to_set_loss import SetToSetLossKinematics, SetToSetLossIncidence
from utility.transformation import VarTransformation

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
plt.style.use('seaborn-v0_8-whitegrid')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix

from pytorch_lightning.utilities import grad_norm


class PflowLightning(LightningModule):

    def __init__(self, config_mv, config_t, comet_logger=None, inference=False):
        super().__init__()
        self.save_hyperparameters()
        self.config_mv = config_mv
        self.config_t = config_t

        self.net = SAPF(self.config_mv['pf_model'], inference=inference)

        # pick the loss function
        if self.config_t.get('loss_on_inc_wts', False):
            self.set_to_set_loss = SetToSetLossIncidence(
                self.config_t, max_part=self.config_mv['pf_model']['max_particles'])
        else:
            self.set_to_set_loss = SetToSetLossKinematics(
                self.config_t, max_part=self.config_mv['pf_model']['max_particles'])

        self.comet_logger = comet_logger
        self.cmap = LinearSegmentedColormap.from_list('custom_cmap', ['cornflowerblue', 'red'])
        self.cmap.set_under('white')

        self.transform_dicts = {}
        for k, v in self.config_mv['var_transform'].items():
            self.transform_dicts[k] = VarTransformation(v)
        
        kin_pred_cfg = self.config_mv['pf_model'].get('kinematics_predictor', {})
        if kin_pred_cfg.get('use_attn_kinematics', False):
            self.net.kinematics_predictor.kin_net.set_trans_dicts(self.transform_dicts)

        self.automatic_optimization = False
        self.reset_kin_plot_dict()


    def reset_kin_plot_dict(self):
        self.kin_plot_dict = {}
        for v in ['pt_raw', 'eta_raw', 'phi', 'e_raw']:
            self.kin_plot_dict[f'truth_{v}'] = []
            self.kin_plot_dict[f'pred_{v}']  = []


    def set_comet_logger(self, comet_logger):
        self.comet_logger = comet_logger


    def forward(self, input):
        return self.net(input)


    def train_dataloader(self):
        ds = PflowDataset(self.config_t['train_glob_arg'], config_mv=self.config_mv,
            energy_threshold=self.config_t['energy_threshold'],
            reduce_ds=self.config_t['reduce_ds_train'], res=self.config_t['resolution'],
            drop_single_part_events=self.config_t.get('drop_single_part_events', False),
            load_incidence=self.config_t.get('loss_on_inc_wts', False))
        
        if self.config_t.get('use_sampler', False):    
            batch_sampler = SuperResSampler(np.array(ds.cell_count),
                batch_size=self.config_t['batch_size_train'], 
                n_sq_sum_threshold=self.config_t['n_sq_sum_threshold_train'], drop_last=False)
            loader = DataLoader(ds, num_workers=self.config_t["num_workers"],
                batch_sampler=batch_sampler, pin_memory=True,
                collate_fn=lambda x: collate_fn(x, self.config_mv['pf_model']['max_particles']))
        else:
            loader = DataLoader(ds, batch_size=self.config_t['batch_size_train'], 
                num_workers=self.config_t['num_workers'], shuffle=True,
                collate_fn=lambda x: collate_fn(x, self.config_mv['pf_model']['max_particles']))

        return loader
    

    def val_dataloader(self):
        ds = PflowDataset(self.config_t['val_glob_arg'], config_mv=self.config_mv, 
            energy_threshold=self.config_t['energy_threshold'],
            reduce_ds=self.config_t['reduce_ds_val'], res=self.config_t['resolution'],
            drop_single_part_events=self.config_t.get('drop_single_part_events', False),
            load_incidence=self.config_t.get('loss_on_inc_wts', False))
        
        if self.config_t.get('use_sampler', False):    
            batch_sampler = SuperResSampler(np.array(ds.cell_count),
                batch_size=self.config_t['batch_size_val'], 
                n_sq_sum_threshold=self.config_t['n_sq_sum_threshold_val'], drop_last=False)
            loader = DataLoader(ds, num_workers=self.config_t["num_workers"],
                batch_sampler=batch_sampler, pin_memory=True,
                collate_fn=lambda x: collate_fn(x, self.config_mv['pf_model']['max_particles']))
        else:
            loader = DataLoader(ds, batch_size=self.config_t['batch_size_val'],
                num_workers=self.config_t['num_workers'], shuffle=False,
                collate_fn=lambda x: collate_fn(x, self.config_mv['pf_model']['max_particles']))

        return loader
    

    def compute_loss(self, pred, batch):
        loss_to_optimize_on = 0
        dict_to_log = {
            'loss': 0, 'card_loss': 0, 'inc_loss': 0, 'kin_loss': 0, 
            'pt_loss': 0, 'eta_loss': 0, 'phi_loss': 0, 'e_loss': 0
        }

        card_pred_logits, kin_pred, inc_weights = pred

        if card_pred_logits != None:
            truth_card = batch['cardinality']
            card_loss = F.cross_entropy(card_pred_logits, truth_card)
            card_loss = self.config_t['card_loss_weight'] * card_loss
            
            loss_to_optimize_on += card_loss
            dict_to_log['card_loss'] = card_loss.item()

        indices = None
        if kin_pred != None:

            if self.config_t.get('loss_on_inc_wts', False):
                inc_loss, loss_components, indices = \
                    self.set_to_set_loss.compute(inc_weights, batch, kin_pred)
                loss_to_optimize_on += inc_loss
                dict_to_log['inc_loss'] = inc_loss.item()

            else:
                kin_loss, loss_components, indices = \
                    self.set_to_set_loss.compute(kin_pred, batch)
                loss_to_optimize_on += kin_loss
                dict_to_log['kin_loss'] = kin_loss.item()

            for k, v in loss_components.items():
                dict_to_log[k] = v

        return loss_to_optimize_on, dict_to_log, indices


    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        bs = batch['cell_mask'].shape[0]
        pred = self.net(batch)
        loss_to_optimize_on, dict_to_log, _ = self.compute_loss(pred, batch)

        # manual optimization
        self.manual_backward(loss_to_optimize_on)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        opt.step()

        log_dict = {}
        for k, v in dict_to_log.items():
            log_dict[f'train/{k}'] = v
        self.log_dict(log_dict, batch_size=bs)

        return loss_to_optimize_on


    def validation_step(self, batch, batch_idx):
        bs = batch['cell_mask'].shape[0]
        pred = self.net(batch)
        card_pred_logits, kin_pred, inc_weights = pred

        loss_to_optimize_on, dict_to_log, indices = self.compute_loss(pred, batch)

        log_dict = {}
        for k, v in dict_to_log.items():
            log_dict[f'val/{k}'] = v
        log_dict['val_loss_to_optimize_on'] = loss_to_optimize_on.item()

        card_plot_dict = {}
        if card_pred_logits != None:
            card_plot_dict['card_truth'] = batch['cardinality'].cpu().numpy()
            card_plot_dict['card_pred']  = card_pred_logits.argmax(-1).detach().cpu().numpy()

        if kin_pred != None:
            for bs_i in range(bs):
                mask_bsi = batch['part_mask'][bs_i]
                self.kin_plot_dict['truth_pt_raw'].append(
                    batch['part_pt_raw'][bs_i, mask_bsi].cpu().numpy())
                self.kin_plot_dict['truth_eta_raw'].append(
                    batch['part_eta_raw'][bs_i, mask_bsi].cpu().numpy())
                self.kin_plot_dict['truth_phi'].append(
                    batch['part_phi'][bs_i, mask_bsi].cpu().numpy())
                self.kin_plot_dict['truth_e_raw'].append(
                    batch['part_e_raw'][bs_i, mask_bsi].cpu().numpy())
                
                ind_bsi = indices[bs_i]
                self.kin_plot_dict['pred_pt_raw'].append(
                    self.transform_dicts['pt'].inverse(kin_pred[bs_i, ind_bsi, 0][mask_bsi]).detach().cpu().numpy())
                self.kin_plot_dict['pred_eta_raw'].append(
                    self.transform_dicts['eta'].inverse(kin_pred[bs_i, ind_bsi, 1][mask_bsi]).detach().cpu().numpy())
                self.kin_plot_dict['pred_phi'].append(
                    kin_pred[bs_i, ind_bsi, 2][mask_bsi].detach().cpu().numpy())
                self.kin_plot_dict['pred_e_raw'].append(
                    self.transform_dicts['e'].inverse(kin_pred[bs_i, ind_bsi, 3][mask_bsi]).detach().cpu().numpy())
                
        return log_dict, card_plot_dict


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

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        norms = grad_norm(self.net, norm_type=2)
        norms2store = {
            'grad_2.0_norm_total': norms['grad_2.0_norm_total']
        }
        self.log_dict(norms2store)


    def training_epoch_end(self, outputs):
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        self.lr_schedulers().step()


    def validation_epoch_end(self, outputs):
        epoch_end_log_dict = {}
        for key in outputs[0][0].keys():
            epoch_end_log_dict[key] = np.hstack([x[0][key] for x in outputs]).mean().item()
        self.log_dict(epoch_end_log_dict)

        # plotting cardinality
        if 'card_truth' in outputs[0][1]:
            truth_card = np.hstack([x[1]['card_truth'] for x in outputs])
            pred_card = np.hstack([x[1]['card_pred'] for x in outputs])
            fig = self.plot_perf_card(truth_card, pred_card)
            self.log_image(fig, 'cardinality')

        # plotting kinematics
        if len(self.kin_plot_dict['truth_pt_raw']) > 0:
            fig = self.plot_perf_kinematics()
            self.log_image(fig, 'kinematics')
            self.reset_kin_plot_dict()
            

    def plot_perf_card(self, truth, pred):
        class_labels = [str(i) for i in range(self.config_mv['pf_model']['max_particles'])]
        n_class = len(class_labels)

        fig = plt.figure(figsize=(9, 6), dpi=100, tight_layout=True)
        ax = fig.add_subplot(111)
        cm = confusion_matrix(pred, truth, labels=np.arange(n_class), normalize=None)
        df_cm = pd.DataFrame(cm, 
            index = [class_labels[i] for i in range(n_class)], 
            columns = [class_labels[i] for i in range(n_class)]
        )
        sn.heatmap(df_cm, annot=True, ax=ax, cmap=self.cmap, vmin=1e-8)
        ax.set_xlabel('truth cardinality')
        ax.set_ylabel('pred cardinality')

        return fig


    def plot_perf_kinematics(self):
        for k, v in self.kin_plot_dict.items():
            self.kin_plot_dict[k] = np.hstack(v)

        # phi to [-pi, pi]
        self.kin_plot_dict['pred_phi'] = (self.kin_plot_dict['pred_phi'] + np.pi) % (2 * np.pi) - np.pi
            
        fig = plt.figure(figsize=(12, 6), dpi=100, tight_layout=True)
        gs = fig.add_gridspec(2, 4, hspace=0.6, wspace=0.3)

        for i, v in enumerate(['pt_raw', 'eta_raw', 'phi', 'e_raw']):
            res = self.kin_plot_dict[f'truth_{v}'] - self.kin_plot_dict[f'pred_{v}']
            rel_res = res / self.kin_plot_dict[f'truth_{v}']

            ax1 = fig.add_subplot(gs[0, i])
            bins = np.linspace(np.percentile(res, 3), np.percentile(res, 97), 100)
            ax1.hist(res, bins=bins, histtype='stepfilled', color='cornflowerblue', ec='k', lw=0.5)
            ax1.set_title(self.get_label(res), fontsize=10)
            ax1.set_xlabel(f'{v} (truth - pred)')

            ax2 = fig.add_subplot(gs[1, i])
            bins = np.linspace(np.percentile(rel_res, 3), np.percentile(rel_res, 97), 100)
            ax2.hist(rel_res, bins=bins, histtype='stepfilled', color='cornflowerblue', ec='k', lw=0.5)
            ax2.set_title(self.get_label(rel_res), fontsize=10)
            ax2.set_xlabel(f'{v} (truth - pred) / truth')

            for ax in [ax1, ax2]:
                ax.grid(True)

        return fig


    def get_label(self, x):
        text = f'$\mu$={x.mean():.2f}, $\sigma$={x.std():.2f}'
        iqr = np.percentile(x, 75) - np.percentile(x, 25)
        text += f'\nmed={np.median(x):.2f}, IQR={iqr:.2f}'
        return text
    

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
import sys
paths = sys.path
for p in paths:
    if '.local' in p:
        paths.remove(p)



# early argparse for CUDA
import argparse, yaml, os
argparser = argparse.ArgumentParser()
argparser.add_argument('--inference_path', '-i', type=str, required=True)
args = argparser.parse_args()

with open(args.inference_path, 'r') as fp:
    inference_cfg_tmp = yaml.safe_load(fp)

# set the gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(inference_cfg_tmp['gpu'])



import uproot
import awkward as ak

from tqdm import tqdm
from pathlib import Path

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from pflow.dataset_pf import PflowDataset, collate_fn
from pflow.lightning_pf import PflowLightning
from utility.sampler import SuperResSampler
from utility.set_to_set_loss import SetToSetLossKinematics, SetToSetLossIncidence
from utility.transformation import VarTransformation

# torch.set_float32_matmul_precision(inference_cfg_tmp['precision'])



class Inference:
    def __init__(self, inference_cfg):
        self.inference_cfg = inference_cfg

        self.config_path_mv = inference_cfg['model']['config_path_mv']
        with open(inference_cfg['model']['config_path_mv'], 'r') as fp:
            self.config_mv = yaml.safe_load(fp)

        self.config_path_t = inference_cfg['model']['config_path_t']
        with open(inference_cfg['model']['config_path_t'], 'r') as fp:
            self.config_t = yaml.safe_load(fp)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.load_model()

        # pick the loss function
        if self.config_t.get('loss_on_inc_wts', False):
            self.set_to_set_loss = SetToSetLossIncidence(
                self.config_t, max_part=self.config_mv['pf_model']['max_particles'])
        else:
            self.set_to_set_loss = SetToSetLossKinematics(
                self.config_t, max_part=self.config_mv['pf_model']['max_particles'])

        self.transform_dicts = {}
        for k, v in self.config_mv['var_transform'].items():
            self.transform_dicts[k] = VarTransformation(v)


    def load_model(self):
        self.lightning_model = PflowLightning(self.config_mv, self.config_t, inference=True)

        checkpoint_path = self.inference_cfg['model']['checkpoint_path']
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.lightning_model.load_state_dict(checkpoint['state_dict'])

        torch.set_grad_enabled(False)
        self.lightning_model.eval()
        self.lightning_model.cuda() if torch.cuda.is_available() else self.lightning_model.cpu()


    def get_dataloader(self, inf_dict):
        ds = PflowDataset(inf_dict['glob_arg'], config_mv=self.config_mv, 
            energy_threshold=self.config_t['energy_threshold'],
            reduce_ds=inf_dict['reduce_ds'], res=self.config_t['resolution'],
            load_incidence=self.config_t.get('loss_on_inc_wts', False))
    
        if self.inference_cfg.get('use_sampler', False):    
            batch_sampler = SuperResSampler(np.array(ds.cell_count),
                batch_size=self.inference_cfg['batch_size'],
                n_sq_sum_threshold=self.config_t['n_sq_sum_threshold_val'], drop_last=False)
            loader = DataLoader(ds, num_workers=self.inference_cfg["num_workers"],
                batch_sampler=batch_sampler, pin_memory=True,
                collate_fn=lambda x: collate_fn(x, self.config_mv['pf_model']['max_particles']))
        else:
            loader = DataLoader(ds, batch_size=self.inference_cfg['batch_size_val'],
                num_workers=self.inference_cfg['num_workers'], shuffle=False,
                collate_fn=lambda x: collate_fn(x, self.config_mv['pf_model']['max_particles']))

        return loader


    def prep_dicts(self, inf_dict):
        self.kin_dict_to_zip = {
            "truth_pt_raw": [], "truth_eta_raw": [], "truth_phi": [], "truth_e_raw": [],
            "truth_dep_e_raw": [],
            "pred_pt_raw": [],  "pred_eta_raw": [],  "pred_phi": [],  "pred_e_raw": []}

        self.card_dict_to_zip = {
            "truth": [], "pred": [], "idx": []}

        if inf_dict.get('store_inc_wt', False):
            self.cell_dict_to_zip = {}
            for i in range(self.config_mv['pf_model']['max_particles']):
                self.cell_dict_to_zip[f"pred_inc_wt_{i}"] = []


    def run_pred(self, inf_dict):
        self.prep_dicts(inf_dict)

        loader = self.get_dataloader(inf_dict)

        # run the predictions
        for batch in tqdm(loader, desc='predicting...', total=len(loader)):

            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(self.device)

            card_pred_logits, kin_pred, inc_weights = self.lightning_model.net(batch)

            # cardinality
            card_pred = np.zeros(batch['cell_mask'].shape[0])
            if card_pred_logits != None:
                card_pred = torch.argmax(card_pred_logits, dim=-1).cpu().numpy()
            self.card_dict_to_zip['truth'].append(batch['cardinality'].cpu().numpy())
            self.card_dict_to_zip['pred'].append(card_pred)
            self.card_dict_to_zip['idx'].append(batch['idx'].cpu().numpy())

            # kinematics
            if self.config_t.get('loss_on_inc_wts', False):
                _, _, indices = self.set_to_set_loss.compute(inc_weights, batch, kin_pred)
            else:
                _, _, indices = self.set_to_set_loss.compute(kin_pred, batch)


            bs = batch['cell_mask'].shape[0]

            for bs_i in range(bs):
                mask_bsi = batch['part_mask'][bs_i]
                self.kin_dict_to_zip['truth_pt_raw'].append(
                    batch['part_pt_raw'][bs_i, mask_bsi].cpu().numpy())
                self.kin_dict_to_zip['truth_eta_raw'].append(
                    batch['part_eta_raw'][bs_i, mask_bsi].cpu().numpy())
                self.kin_dict_to_zip['truth_phi'].append(
                    batch['part_phi'][bs_i, mask_bsi].cpu().numpy())
                self.kin_dict_to_zip['truth_e_raw'].append(
                    batch['part_e_raw'][bs_i, mask_bsi].cpu().numpy())
                self.kin_dict_to_zip['truth_dep_e_raw'].append(
                    batch['part_dep_e_raw'][bs_i, mask_bsi].cpu().numpy())

                ind_bsi = indices[bs_i]
                self.kin_dict_to_zip['pred_pt_raw'].append(
                    self.transform_dicts['pt'].inverse(kin_pred[bs_i, ind_bsi, 0][mask_bsi]).detach().cpu().numpy())
                self.kin_dict_to_zip['pred_eta_raw'].append(
                    self.transform_dicts['eta'].inverse(kin_pred[bs_i, ind_bsi, 1][mask_bsi]).detach().cpu().numpy())
                self.kin_dict_to_zip['pred_phi'].append(
                    kin_pred[bs_i, ind_bsi, 2][mask_bsi].detach().cpu().numpy())
                self.kin_dict_to_zip['pred_e_raw'].append(
                    self.transform_dicts['e'].inverse(kin_pred[bs_i, ind_bsi, 3][mask_bsi]).detach().cpu().numpy())
                

                if inf_dict.get('store_inc_wt', False):
                    inc_weights_ev_reordered = inc_weights[bs_i, ind_bsi]
                    cell_mask_bsi = batch['cell_mask'][bs_i]
                    for i in range(self.config_mv['pf_model']['max_particles']):
                        self.cell_dict_to_zip[f"pred_inc_wt_{i}"].append(
                            inc_weights_ev_reordered[i, cell_mask_bsi].detach().cpu().numpy())

        self.write_to_root(inf_dict)


    def write_to_root(self, inf_dict):
        for k, v in self.card_dict_to_zip.items():
            self.card_dict_to_zip[k] = np.hstack(v)

        with uproot.recreate(inf_dict['pred_path']) as file:
            _dict = {
                "": ak.zip(self.kin_dict_to_zip),
                "truth_card": self.card_dict_to_zip['truth'],
                "pred_card": self.card_dict_to_zip['pred'],
                "idx": self.card_dict_to_zip['idx']
            }
            if inf_dict.get('store_inc_wt', False):
                _dict["cell"] = ak.zip(self.cell_dict_to_zip)

            file["Particle_Tree"] = _dict

            print('\nParticle_Tree')
            file["Particle_Tree"].show()

        print(f"\nPredictions saved to {inf_dict['pred_path']}")


    def get_output_path(self, inf_dict):
        outputdir = os.path.join(os.path.dirname(self.config_path_mv), 'inference')
        Path(outputdir).mkdir(parents=True, exist_ok=True)
        pred_path = os.path.join(
            outputdir, inf_dict['pred_file_name'])
        
        return pred_path







if __name__ == '__main__':

    with open(args.inference_path, 'r') as fp:
        inference_cfg = yaml.safe_load(fp)

        torch.set_float32_matmul_precision(inference_cfg['precision'])

        inf_obj = Inference(inference_cfg)

        for inf_dict in inference_cfg['items']:
            if inf_dict['pred_path'] is None:
                inf_dict['pred_path'] = inf_obj.get_output_path(inf_dict)
            inf_obj.run_pred(inf_dict)

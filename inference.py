import sys
paths = sys.path
for p in paths:
    if '.local' in p:
        paths.remove(p)

import uproot
import awkward as ak

sys.path.append('./models/')
sys.path.append('./utility/')
import utility.transformation as trans

from tqdm import tqdm
import yaml
import os

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SupResDataset, collate_graphs_plus
from lightning import SupResLightning

from pathlib import Path

import argparse
from performance import PerformanceCOCOA
from utility.target_transformation import TargetTransformation

import time



class Inference:
    def __init__(self, inf_cfg):
        self.inf_cfg = inf_cfg

        self.config_path_mv = inf_cfg['model']['config_path_mv']
        with open(inf_cfg['model']['config_path_mv'], 'r') as fp:
            self.config_mv = yaml.safe_load(fp)

        self.config_path_t = inf_cfg['model']['config_path_t']
        with open(inf_cfg['model']['config_path_t'], 'r') as fp:
            self.config_t = yaml.safe_load(fp)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.load_model()

        n_steps = self.inf_cfg['model']['n_steps']
        time_steps_used = np.linspace(0, 1, n_steps)

        n_steps_to_store = self.inf_cfg['model']['n_steps_to_store']
        time_steps_to_store = np.linspace(0, 1, n_steps_to_store+1)

        # store intermediate results
        self.ts_to_store = []
        self.ts_to_store_idx = []
        for t in time_steps_to_store:
            idx = np.argmin(np.abs(time_steps_used - t))
            self.ts_to_store.append(time_steps_used[idx])
            self.ts_to_store_idx.append(idx)

        self.ts_to_store = self.ts_to_store[:-1]
        self.ts_to_store_idx = self.ts_to_store_idx[:-1]

        self.target_trans_obj = TargetTransformation(self.config_mv['target_transform'])


    def load_model(self):
        self.lightning_model = SupResLightning(self.config_mv, self.config_t)

        checkpoint_path = self.inf_cfg['model']['checkpoint_path']
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.lightning_model.load_state_dict(checkpoint['state_dict'])

        torch.set_grad_enabled(False)
        self.lightning_model.eval()
        self.lightning_model.cuda() if torch.cuda.is_available() else self.lightning_model.cpu()


    def get_dataloader(self, inf_dict):
        reduce_ds = inf_dict['n_events']
        ds = SupResDataset(inf_dict['truth_path'], reduce_ds=reduce_ds, entry_start=inf_dict['entry_start'],
            config_mv=self.config_mv, make_low_graph=True, make_particle_graph=True,
            one_event_train=self.config_t['one_event_train'], one_event_idx=self.config_t['one_event_idx'])
        loader = DataLoader(ds, batch_size=inf_dict['batch_size'], 
            num_workers=inf_dict['num_workers'], shuffle=False, collate_fn=collate_graphs_plus)
        return loader


    def prep_dicts(self, inf_dict):
        # create disctionaries. to be used for writing to root
        self.low_dict_to_zip = {
            "eta_raw": [], "phi": [], "layer": [], "e_meas_raw": []}

        self.high_dict_to_zip = {
            "eta_raw": [], "phi": [], "layer": [], 
            "e_proxy": [], "e_truth_raw": [], "e_proxy_raw": [],
            "e_pred_raw": [], "e_pred_avg_raw": [], # avg(unscale(pred)) vs unscale(avg(pred))
            "raw_nn_cond": [], "raw_nn_target": [], "raw_nn_pred": []}
        for i in self.ts_to_store:
            self.high_dict_to_zip[f"e_pred_raw_{i:.2f}"] = []
            self.high_dict_to_zip[f"e_pred_avg_raw_{i:.2f}"] = []
            self.high_dict_to_zip[f"raw_nn_pred_{i:.2f}"] = []

        # ensemble mode
        if inf_dict.get('n_ensemble', 1) > 1 and inf_dict['store_ensemble_components']:
            for i in range(inf_dict['n_ensemble']):
                self.high_dict_to_zip[f"e_pred_raw_comp_{i}"] = []
                self.high_dict_to_zip[f"raw_nn_pred_comp_{i}"] = []

                for t in self.ts_to_store:
                    self.high_dict_to_zip[f"e_pred_raw_{t:.2f}_comp_{i}"] = []
                    self.high_dict_to_zip[f"raw_nn_pred_{t:.2f}_comp_{i}"] = []

        # particles
        self.particle_dict_to_zip = {
            "particle_pt": [], "particle_eta": [], "particle_phi": [], 
            "particle_e": [], "particle_pdgid": [], "particle_dep_e": []}

        # energy incidence
        if inf_dict['store_energy_incidence']:
            for i in range(inf_dict['max_particles']):
                self.low_dict_to_zip[f"e_part_{i}"] = []
                self.high_dict_to_zip[f"e_part_{i}"] = []


    def run_pred(self, inf_dict):
        self.prep_dicts(inf_dict)

        loader = self.get_dataloader(inf_dict)

        # run the predictions
        for batch in tqdm(loader, desc='predicting...', total=len(loader)):

            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(self.device)

            pred_comp_list = []
            for i in range(inf_dict.get('n_ensemble', 1)):
                pred_comp = self.lightning_model.net.generate_samples(
                    batch, n_steps=inf_dict['n_steps'], ret_seq=True)
                pred_comp_list.append(pred_comp)

            # ensemble average
            pred_avg = torch.stack(pred_comp_list, dim=0).mean(dim=0)

            self.fill_the_dicts2write(
                batch, pred_avg,
                pred_comp_list if inf_dict.get('n_ensemble', 1) > 1 else None) # we need to pass the ensemble components to compute the avg
                
            # break ##### HACK ####

        self.write_to_root(inf_dict['pred_path'])


    def fill_the_dicts2write(self, batch, pred_avg, pred_comp_list=None):
        '''
            enseble avg can be computed in two ways
            1. avg(nn outputs) -> unscale
                "e_pred_avg_raw"
            2. unscale(nn outputs) -> avg
                "e_pred_raw"  # this one seems better

            we have -
                e_pred_abg_raw, e_pred_avg_raw_t


        '''

        batch_size = batch['q_mask'].shape[0]

        for bs_i in range(batch_size):

            # less code. not most optimal performance
            tmp_dict_low = {}; tmp_dict_high = {}

            low_q_mask_bs_i = batch['low_q_mask'][bs_i]
            high_q_mask_bs_i = batch['q_mask'][bs_i]

            tmp_dict_low["eta_raw"]    = batch['low_eta_raw'][bs_i][low_q_mask_bs_i]
            tmp_dict_low["phi"]        = batch['low_phi'][bs_i][low_q_mask_bs_i]
            tmp_dict_low["layer"]      = batch['low_layer'][bs_i][low_q_mask_bs_i]
            tmp_dict_low["e_meas_raw"] = batch['low_e_meas_raw'][bs_i][low_q_mask_bs_i] * 1e3

            tmp_dict_high["eta_raw"]     = batch['eta_raw'][bs_i][high_q_mask_bs_i]
            tmp_dict_high["phi"]         = batch['phi'][bs_i][high_q_mask_bs_i]
            tmp_dict_high["layer"]       = batch['layer'][bs_i][high_q_mask_bs_i]
            tmp_dict_high["e_truth_raw"] = batch['e_truth_raw'][bs_i][high_q_mask_bs_i] * 1e3
            tmp_dict_high["e_proxy"]     = batch['e_proxy'][bs_i][high_q_mask_bs_i]
            tmp_dict_high["e_proxy_raw"] = batch['e_proxy_raw'][bs_i][high_q_mask_bs_i] * 1e3

            e_pred_raw_bs_i = self.target_trans_obj.inverse(
                pred_avg[-1, bs_i, ...][high_q_mask_bs_i], batch['e_proxy_raw'][bs_i][high_q_mask_bs_i])
            tmp_dict_high["e_pred_avg_raw"]    = e_pred_raw_bs_i * 1e3

            # will replace this later, if ensemble
            tmp_dict_high["e_pred_raw"] = e_pred_raw_bs_i * 1e3

            tmp_dict_high["raw_nn_cond"]   = batch['e_proxy'][bs_i][high_q_mask_bs_i]
            tmp_dict_high["raw_nn_target"] = batch['target'][bs_i][high_q_mask_bs_i]
            tmp_dict_high["raw_nn_pred"]   = pred_avg[-1, bs_i, ...][high_q_mask_bs_i]

            for t, ts_i in zip(self.ts_to_store, self.ts_to_store_idx):
                e_pred_raw_tmp = self.target_trans_obj.inverse(
                    pred_avg[ts_i, bs_i, ...][high_q_mask_bs_i], batch['e_proxy_raw'][bs_i][high_q_mask_bs_i])
                tmp_dict_high[f"e_pred_avg_raw_{t:.2f}"] = e_pred_raw_tmp * 1e3
                tmp_dict_high[f"raw_nn_pred_{t:.2f}"] = pred_avg[ts_i, bs_i, ...][high_q_mask_bs_i]
                
                # will replace this later, if ensemble
                tmp_dict_high[f"e_pred_raw_{t:.2f}"] = e_pred_raw_tmp * 1e3


            # ensemble components (if one component, pred_comp_list is None)
            if pred_comp_list != None:

                tmp_dict_high["e_pred_raw"] = torch.zeros_like(e_pred_raw_bs_i)
                for t, ts_i in zip(self.ts_to_store, self.ts_to_store_idx):
                    tmp_dict_high[f"e_pred_raw_{t:.2f}"] = torch.zeros_like(e_pred_raw_bs_i)
                
                # loop over ensemble components
                for i, pred_comp in enumerate(pred_comp_list):
                    e_pred_raw_comp_i_bs_i = self.target_trans_obj.inverse(
                        pred_comp[-1, bs_i, ...][high_q_mask_bs_i], batch['e_proxy_raw'][bs_i][high_q_mask_bs_i])
                    

                    # STORE ONLY IF NECESSARY
                    if 'e_pred_raw_comp_0' in self.high_dict_to_zip.keys():
                        tmp_dict_high[f"e_pred_raw_comp_{i}"] = e_pred_raw_comp_i_bs_i * 1e3
                        tmp_dict_high[f"raw_nn_pred_comp_{i}"] = pred_comp[-1, bs_i, ...][high_q_mask_bs_i]

                    # ensemble sum
                    tmp_dict_high[f"e_pred_raw"] += e_pred_raw_comp_i_bs_i * 1e3


                    # loop over time steps to store
                    for t, ts_i in zip(self.ts_to_store, self.ts_to_store_idx):
                        e_pred_raw_tmp = self.target_trans_obj.inverse(
                            pred_comp[ts_i, bs_i, ...][high_q_mask_bs_i], batch['e_proxy_raw'][bs_i][high_q_mask_bs_i])

                        # STORE ONLY IF NECESSARY
                        if 'e_pred_raw_comp_0' in self.high_dict_to_zip.keys():
                            tmp_dict_high[f"e_pred_raw_{t:.2f}_comp_{i}"] = e_pred_raw_tmp * 1e3
                            tmp_dict_high[f"raw_nn_pred_{t:.2f}_comp_{i}"] = pred_comp[ts_i, bs_i, ...][high_q_mask_bs_i]

                        # ensemble sum
                        tmp_dict_high[f"e_pred_raw_{t:.2f}"] += e_pred_raw_tmp * 1e3


                # ensemble average
                tmp_dict_high["e_pred_raw"] /= len(pred_comp_list)
                for t, ts_i in zip(self.ts_to_store, self.ts_to_store_idx):
                    tmp_dict_high[f"e_pred_raw_{t:.2f}"] /= len(pred_comp_list)


            # energy incidence matrix
            if inf_dict['store_energy_incidence']:
                n_part = batch['particle_pt'][bs_i].shape[0]
                for pi in range(n_part):
                    tmp_dict_low[f"e_part_{pi}"] = batch[f'low_e_part_{pi}'][bs_i]
                    tmp_dict_high[f"e_part_{pi}"] = batch[f'high_e_part_{pi}'][bs_i]
                for pi in range(n_part, inf_dict['max_particles']):
                    tmp_dict_low[f"e_part_{pi}"] = torch.zeros_like(batch['low_e_part_0'][bs_i])
                    tmp_dict_high[f"e_part_{pi}"] = torch.zeros_like(batch['high_e_part_0'][bs_i])




            for k, v in tmp_dict_low.items():
                self.low_dict_to_zip[k].append(v.squeeze(-1).detach().cpu().numpy())
            for k, v in tmp_dict_high.items():
                self.high_dict_to_zip[k].append(v.squeeze(-1).detach().cpu().numpy())


            # particles (list of tensors; no padding)
            self.particle_dict_to_zip["particle_pt"].append(batch['particle_pt'][bs_i].detach().cpu().numpy())
            self.particle_dict_to_zip["particle_eta"].append(batch['particle_eta'][bs_i].detach().cpu().numpy())
            self.particle_dict_to_zip["particle_phi"].append(batch['particle_phi'][bs_i].detach().cpu().numpy())
            self.particle_dict_to_zip["particle_e"].append(batch['particle_e'][bs_i].detach().cpu().numpy())
            self.particle_dict_to_zip["particle_pdgid"].append(batch['particle_pdgid'][bs_i].detach().cpu().numpy())
            self.particle_dict_to_zip["particle_dep_e"].append(batch['particle_dep_e'][bs_i].detach().cpu().numpy())



    def write_to_root(self, pred_path):
        with uproot.recreate(pred_path) as file:
            file["Low_Tree"] = {
                "": ak.zip(self.low_dict_to_zip)
            }
            file["High_Tree"] = {
                "": ak.zip(self.high_dict_to_zip)
            }
            file["Particle_Tree"] = {
                "": ak.zip(self.particle_dict_to_zip)
            }

            print('\nLow_Tree')
            file["Low_Tree"].show()
            print('\nHigh_Tree')
            file["High_Tree"].show()
            print('\nParticle_Tree')
            file["Particle_Tree"].show()

        print(f"\nPredictions saved to {pred_path}")


    def get_output_path(self, inf_dict):
        # create the directory
        outputdir = os.path.join(os.path.dirname(self.config_path_mv), 'inference')
        if inf_dict.get('dir_flag', None) != None:
            outputdir = os.path.join(outputdir, inf_dict['dir_flag'])
        Path(outputdir).mkdir(parents=True, exist_ok=True)

        pred_path = os.path.join(
            outputdir, '{}_pred.root'.format(
                '_'.join(inf_dict['truth_path'].split('.root')[0].split('/')[-1:])))
        
        return pred_path


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--inference_path', '-i', type=str, required=True)
    argparser.add_argument('--precision', '-p', type=str, required=False, default='highest')
    argparser.add_argument('--batch_mode', '-bm', action='store_true')
    argparser.add_argument('--entry_start', '-estart', type=int, required=False, default=0)
    argparser.add_argument('--entry_stop', '-estop', type=int, required=False, default=None)
    args = argparser.parse_args()

    with open(args.inference_path, 'r') as fp:
        inference_cfg = yaml.safe_load(fp)


    # if batch mode, pass the batch mode config
    if args.batch_mode:
        assert 'items' not in inference_cfg, 'wrong config style for batch mode'
        assert 'entry_stop' != None, 'entry_stop is required for batch mode'

        # set the precision and GPU
        torch.set_float32_matmul_precision(args.precision)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(inference_cfg['gpu'])

        inf_obj = Inference(inference_cfg)

        inf_dict = inference_cfg['inf_dict']
        inf_dict['entry_start'] = args.entry_start
        inf_dict['n_events'] = args.entry_stop - args.entry_start
        inf_dict['num_workers'] = inference_cfg['num_workers']
        inf_dict['batch_size'] = inference_cfg['batch_size']
        inf_dict['n_steps'] = inference_cfg['model']['n_steps']
        inf_dict['n_steps_to_store'] = inference_cfg['model']['n_steps_to_store']
        inf_dict['max_particles'] = inference_cfg['max_particles']

        pred_path_tmp = inf_obj.get_output_path(inf_dict)
        pred_path = pred_path_tmp.replace('.root', f'_{args.entry_start}_{args.entry_stop}.root')
        inf_dict['pred_path'] = pred_path

        t1 = time.time()
        inf_obj.run_pred(inf_dict)
        t2 = time.time()
        print(f'Prediction time: {t2-t1:.2f} s')

    else:
        assert 'items' in inference_cfg, 'wrong config style for not batch mode'

        # set the gpu, precision
        os.environ["CUDA_VISIBLE_DEVICES"] = str(inference_cfg['gpu'])
        torch.set_float32_matmul_precision(args.precision)

        inf_obj = Inference(inference_cfg)

        
        for inf_dict in inference_cfg['items']:

            if inf_dict['run_pred']:
                print('Running predictions on {}'.format(inf_dict['truth_path']))

                inf_dict['gpu'] = inference_cfg['gpu']
                inf_dict['num_workers'] = inference_cfg['num_workers']
                inf_dict['batch_size'] = inference_cfg['batch_size']
                inf_dict['n_steps'] = inference_cfg['model']['n_steps']
                inf_dict['n_steps_to_store'] = inference_cfg['model']['n_steps_to_store']
                inf_dict['pred_path'] = inf_obj.get_output_path(inf_dict)
                inf_dict['max_particles'] = inference_cfg['max_particles']

                inf_obj.run_pred(inf_dict)

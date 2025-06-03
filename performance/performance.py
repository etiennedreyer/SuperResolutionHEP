import uproot
import numpy as np
from tqdm import tqdm


class PerformanceCOCOA:
    
    def __init__(self, inference_path, res_factor, cmap='viridis', entry_stop=None, max_comp=-1):

        f_inference = uproot.open(inference_path)

        self.res_factor = res_factor

        self.high_gran = [256, 256, 128, 64, 64, 32]
        if res_factor == 2:
            self.low_gran  = [128, 128,  64, 32, 32, 16]
        elif res_factor == 4:
            self.low_gran  = [ 64,  64,  32, 16, 16,  8]
        else:
            raise ValueError('res_factor must be 2 or 4')

        tree_low  = f_inference['Low_Tree']
        tree_high = f_inference['High_Tree']

        self.n_events = tree_low.num_entries
        if entry_stop is not None:
            self.n_events = min(self.n_events, entry_stop)
        entry_stop = self.n_events

        self.low_phi        = tree_low['phi'].array(library='np', entry_start=0, entry_stop=entry_stop)
        self.low_layer      = tree_low['layer'].array(library='np', entry_start=0, entry_stop=entry_stop)
        self.low_eta        = tree_low['eta_raw'].array(library='np', entry_start=0, entry_stop=entry_stop)
        self.low_e_measured = tree_low['e_meas_raw'].array(library='np', entry_start=0, entry_stop=entry_stop)
    
        self.high_phi       = tree_high['phi'].array(library='np', entry_start=0, entry_stop=entry_stop)
        self.high_layer     = tree_high['layer'].array(library='np', entry_start=0, entry_stop=entry_stop)
        self.high_eta       = tree_high['eta_raw'].array(library='np', entry_start=0, entry_stop=entry_stop)

        # MeV
        self.high_e_truth   = tree_high['e_truth_raw'].array(library='np', entry_start=0, entry_stop=entry_stop)
        self.high_e_pred_direct = tree_high['e_pred_raw'].array(library='np', entry_start=0, entry_stop=entry_stop)
        self.high_e_proxy   = tree_high['e_proxy_raw'].array(library='np', entry_start=0, entry_stop=entry_stop)

        self.high_raw_nn_cond   = tree_high['raw_nn_cond'].array(library='np', entry_start=0, entry_stop=entry_stop)
        self.high_raw_nn_target = tree_high['raw_nn_target'].array(library='np', entry_start=0, entry_stop=entry_stop)
        self.high_raw_nn_pred   = tree_high['raw_nn_pred'].array(library='np', entry_start=0, entry_stop=entry_stop)

        self.high_e_pred_step = {}; self.high_raw_nn_pred_step = {}
        for br in tqdm(tree_high.keys(), desc='loading step-wise predictions', total=len(tree_high.keys())):
            if 'e_pred_raw_' in br and 'comp' not in br:
                self.high_e_pred_step[br] = tree_high[br].array(library='np', entry_start=0, entry_stop=entry_stop)
            if 'raw_nn_pred_' in br and 'comp' not in br:
                self.high_raw_nn_pred_step[br] = tree_high[br].array(library='np', entry_start=0, entry_stop=entry_stop)

        # the ensemble components
        self.high_e_pred_raw_comp = {}
        for br in tqdm(tree_high.keys(), desc='loading ensemble components', total=len(tree_high.keys())):
            if 'e_pred_raw_comp' in br:
                self.high_e_pred_raw_comp[br] = tree_high[br].array(library='np', entry_start=0, entry_stop=entry_stop)

        # computing the ensemble average
        if len(self.high_e_pred_raw_comp.keys()) == 0:
            self.high_e_pred = self.high_e_pred_direct

        else:
            self.high_e_pred = []
            comp_keys = list(self.high_e_pred_raw_comp.keys())
            denominator = min(len(comp_keys), max_comp) if max_comp > 0 else len(comp_keys)

            for i in tqdm(range(self.n_events), desc='computing ensemble average', total=entry_stop):
                high_e_pred_ev = np.zeros_like(self.high_e_truth[i])
                for comp_i, k in enumerate(comp_keys):
                    high_e_pred_ev += self.high_e_pred_raw_comp[k][i]
                    if max_comp > 0 and comp_i == max_comp:
                        break
                self.high_e_pred.append(high_e_pred_ev / denominator)
            self.high_e_pred = np.array(self.high_e_pred, dtype=object)

        self.cmap = cmap



    def compute_ensemble_average(self, n):
        high_e_pred_avg = []
        comp_keys = list(self.high_e_pred_raw_comp.keys())
        for i in range(self.n_events):
            high_e_pred_ev = np.zeros_like(self.high_e_truth[i])
            for k in comp_keys[:n]:
                high_e_pred_ev += self.high_e_pred_raw_comp[k][i]
            high_e_pred_avg.append(high_e_pred_ev / n)
        high_e_pred_avg = np.array(high_e_pred_avg, dtype=object)

        return high_e_pred_avg
    


class PFPerformanceCOCOA(PerformanceCOCOA):
    def __init__(self, inference_path, lr_pf_path, hr_pf_path, res_factor, cmap='viridis'):
        super().__init__(inference_path, res_factor, cmap)

        # low res
        tree_lr_pf = uproot.open(lr_pf_path)['Particle_Tree']
        self.max_part = len([k for k in tree_lr_pf.keys() if 'cell_pred_inc_wt_' in k])
        n_events = tree_lr_pf.num_entries

        lr_pf_idx_remap = {}
        for i, idx in enumerate(tree_lr_pf['idx'].array(library='np')):
            lr_pf_idx_remap[idx] = i

        self.inc_wt_lr_pf = {}
        for i in range(self.max_part):
            tmp_e_enc_i = tree_lr_pf[f'cell_pred_inc_wt_{i}'].array(library='np')
            self.inc_wt_lr_pf[i] = np.array([
                tmp_e_enc_i[lr_pf_idx_remap[j]] for j in range(n_events)], dtype=object)
            
        # lr truth
        truth_part_pt  = tree_lr_pf['truth_pt_raw'].array(library='np')
        truth_part_eta = tree_lr_pf['truth_eta_raw'].array(library='np')
        truth_part_phi = tree_lr_pf['truth_phi'].array(library='np')
        truth_part_e   = tree_lr_pf['truth_e_raw'].array(library='np')
        truth_part_dep_e = tree_lr_pf['truth_dep_e_raw'].array(library='np')

        self.truth_part_pt = np.array([
            truth_part_pt[lr_pf_idx_remap[j]] for j in range(n_events)], dtype=object)
        self.truth_part_eta = np.array([
            truth_part_eta[lr_pf_idx_remap[j]] for j in range(n_events)], dtype=object)
        self.truth_part_phi = np.array([
            truth_part_phi[lr_pf_idx_remap[j]] for j in range(n_events)], dtype=object)
        self.truth_part_e = np.array([
            truth_part_e[lr_pf_idx_remap[j]] for j in range(n_events)], dtype=object)
        self.truth_part_dep_e = np.array([
            truth_part_dep_e[lr_pf_idx_remap[j]] for j in range(n_events)], dtype=object)

        # lr pred
        low_part_pt  = tree_lr_pf['pred_pt_raw'].array(library='np')
        low_part_eta = tree_lr_pf['pred_eta_raw'].array(library='np')
        low_part_phi = tree_lr_pf['pred_phi'].array(library='np')
        low_part_e   = tree_lr_pf['pred_e_raw'].array(library='np')

        self.low_part_pt = np.array([
            low_part_pt[lr_pf_idx_remap[j]] for j in range(n_events)], dtype=object)
        self.low_part_eta = np.array([
            low_part_eta[lr_pf_idx_remap[j]] for j in range(n_events)], dtype=object)
        self.low_part_phi = np.array([
            low_part_phi[lr_pf_idx_remap[j]] for j in range(n_events)], dtype=object)
        self.low_part_e = np.array([
            low_part_e[lr_pf_idx_remap[j]] for j in range(n_events)], dtype=object)


        # high res
        tree_hr_pf = uproot.open(hr_pf_path)['Particle_Tree']
        max_part = len([k for k in tree_hr_pf.keys() if 'cell_pred_inc_wt_' in k])
        n_events = tree_hr_pf.num_entries

        hr_pf_idx_remap = {}
        for i, idx in enumerate(tree_hr_pf['idx'].array(library='np')):
            hr_pf_idx_remap[idx] = i

        self.inc_wt_hr_pf = {}
        for i in range(self.max_part):
            tmp_e_enc_i = tree_hr_pf[f'cell_pred_inc_wt_{i}'].array(library='np')
            self.inc_wt_hr_pf[i] = np.array([
                tmp_e_enc_i[hr_pf_idx_remap[j]] for j in range(n_events)], dtype=object)

        hr_truth_part_pt  = tree_hr_pf['truth_pt_raw'].array(library='np')
        hr_truth_part_eta = tree_hr_pf['truth_eta_raw'].array(library='np')
        hr_truth_part_phi = tree_hr_pf['truth_phi'].array(library='np')
        hr_truth_part_e   = tree_hr_pf['truth_e_raw'].array(library='np')
        hr_truth_part_dep_e = tree_hr_pf['truth_dep_e_raw'].array(library='np')

        hr_truth_part_pt  = np.array([
            hr_truth_part_pt[hr_pf_idx_remap[j]] for j in range(n_events)], dtype=object)
        hr_truth_part_eta = np.array([
            hr_truth_part_eta[hr_pf_idx_remap[j]] for j in range(n_events)], dtype=object)
        hr_truth_part_phi = np.array([
            hr_truth_part_phi[hr_pf_idx_remap[j]] for j in range(n_events)], dtype=object)
        hr_truth_part_e   = np.array([
            hr_truth_part_e[hr_pf_idx_remap[j]] for j in range(n_events)], dtype=object)
        hr_truth_part_dep_e = np.array([
            hr_truth_part_dep_e[hr_pf_idx_remap[j]] for j in range(n_events)], dtype=object)

        assert np.all(np.hstack(self.truth_part_pt) == np.hstack(hr_truth_part_pt))
        assert np.all(np.hstack(self.truth_part_eta) == np.hstack(hr_truth_part_eta))
        assert np.all(np.hstack(self.truth_part_phi) == np.hstack(hr_truth_part_phi))
        assert np.all(np.hstack(self.truth_part_e) == np.hstack(hr_truth_part_e))
        assert np.all(np.hstack(self.truth_part_dep_e) == np.hstack(hr_truth_part_dep_e))

        high_part_pt  = tree_hr_pf['pred_pt_raw'].array(library='np')
        high_part_eta = tree_hr_pf['pred_eta_raw'].array(library='np')
        high_part_phi = tree_hr_pf['pred_phi'].array(library='np')
        high_part_e   = tree_hr_pf['pred_e_raw'].array(library='np')
        
        self.high_part_pt = np.array([
            high_part_pt[hr_pf_idx_remap[j]] for j in range(n_events)], dtype=object)
        self.high_part_eta = np.array([
            high_part_eta[hr_pf_idx_remap[j]] for j in range(n_events)], dtype=object)
        self.high_part_phi = np.array([
            high_part_phi[hr_pf_idx_remap[j]] for j in range(n_events)], dtype=object)
        self.high_part_e = np.array([
            high_part_e[hr_pf_idx_remap[j]] for j in range(n_events)], dtype=object)


        # particle colors
        colors = [(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228),(56, 34, 132), (160, 194, 56)]
        self.pf_colors = np.array([(r/255, g/255, b/255) for r, g, b in colors]) # normalize to [0, 1]
        self.pf_colors = self.pf_colors[:self.max_part]



        # cardinality (wanted to cehrry-pick an event with a specific cardinality)
        self.truth_cardinality = np.array([len(x) for x in self.truth_part_pt])

        self.low_cardinality = tree_lr_pf['pred_card'].array(library='np')
        self.low_cardinality = np.array([self.low_cardinality[lr_pf_idx_remap[j]] for j in range(n_events)])
        
        self.high_cardinality = tree_hr_pf['pred_card'].array(library='np')
        self.high_cardinality = np.array([self.high_cardinality[hr_pf_idx_remap[j]] for j in range(n_events)])
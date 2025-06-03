import numpy as np
import uproot
import os
import argparse

import energyflow as ef



def calc_substructure(e, eta, phi):
    pt_eta_phis = [
        np.stack([e/np.cosh(eta), eta, phi], axis=-1) for e, eta, phi in zip(e, eta, phi)
    ]

    d2_calc = ef.D2(measure="hadr", beta=1, coords="ptyphim", reg=1e-31)
    c2_calc = ef.C2(measure="hadr", beta=1, coords="ptyphim", reg=1e-31)
    c3_calc = ef.C3(measure="hadr", beta=1, coords="ptyphim", reg=1e-31)

    d2, c2, c3 = [], [], []
    d2 = d2_calc.batch_compute(pt_eta_phis, n_jobs=None)
    c2 = c2_calc.batch_compute(pt_eta_phis, n_jobs=None)
    c3 = c3_calc.batch_compute(pt_eta_phis, n_jobs=None)

    return np.array(d2), np.array(c2), np.array(c3)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--file_path', '-fp', type=str, required=True)
    argparser.add_argument('--entry_start', '-estart', type=int, required=True)
    argparser.add_argument('--entry_stop', '-estop', type=int, required=True)
    argparser.add_argument('--save_dir', '-sd', type=str, required=True)
    args = argparser.parse_args()

    file_path = args.file_path
    entry_start = args.entry_start
    entry_stop = args.entry_stop
    save_dir = args.save_dir

    with uproot.open(file_path) as f:
        tree_low = f['Low_Tree']
        tree_high = f['High_Tree']

        low_phi = tree_low['phi'].array(library='np', entry_start=entry_start, entry_stop=entry_stop)
        low_eta = tree_low['eta_raw'].array(library='np', entry_start=entry_start, entry_stop=entry_stop)
        low_e_measured = tree_low['e_meas_raw'].array(library='np', entry_start=entry_start, entry_stop=entry_stop)

        low_e_split = np.array([x.repeat(4) / 4 for x in low_e_measured], dtype=object)

        high_phi = tree_high['phi'].array(library='np', entry_start=entry_start, entry_stop=entry_stop)
        high_eta = tree_high['eta_raw'].array(library='np', entry_start=entry_start, entry_stop=entry_stop)
        high_e_truth = tree_high['e_truth_raw'].array(library='np', entry_start=entry_start, entry_stop=entry_stop)

        # the ensemble components (ensemble avg in the file is bugged somehow)
        high_e_pred_comp = {}
        for br in tree_high.keys():
            if 'e_pred_raw_comp' in br:
                which_comp = int(br.split('_')[-1])
                if which_comp > 9: # only the first 10 components
                    continue
                high_e_pred_comp[br] = tree_high[br].array(library='np', entry_start=entry_start, entry_stop=entry_stop)

        high_e_pred = []
        comp_keys = list(high_e_pred_comp.keys())
        for i in range(entry_stop - entry_start):
            high_e_pred_ev = np.zeros_like(high_e_truth[i])
            for k in comp_keys:
                high_e_pred_ev += high_e_pred_comp[k][i]
            high_e_pred.append(high_e_pred_ev / len(comp_keys))
        high_e_pred = np.array(high_e_pred, dtype=object)


    d2_low, c2_low, c3_low = calc_substructure(
        low_e_measured, low_eta, low_phi)
    d2_low_split, c2_low_split, c3_low_split = calc_substructure(
        low_e_split, high_eta, high_phi)
    d2_high_truth, c2_high_truth, c3_high_truth = calc_substructure(
        high_e_truth, high_eta, high_phi)
    d2_high_pred, c2_high_pred, c3_high_pred = calc_substructure(
        high_e_pred, high_eta, high_phi)
    
    np.savez(
        os.path.join(save_dir, f'substructures_{entry_start}_{entry_stop}.npz'),
        d2_low=d2_low, c2_low=c2_low, c3_low=c3_low,
        d2_low_split=d2_low_split, c2_low_split=c2_low_split, c3_low_split=c3_low_split,
        d2_high_truth=d2_high_truth, c2_high_truth=c2_high_truth, c3_high_truth=c3_high_truth,
        d2_high_pred=d2_high_pred, c2_high_pred=c2_high_pred, c3_high_pred=c3_high_pred
    )
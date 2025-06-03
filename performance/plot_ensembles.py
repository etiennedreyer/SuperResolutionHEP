import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

from collections import OrderedDict
import os



def plot_residual_event_ens(self, dir=None, truth_e_range=None):

    dict_e_layer = {
        'low_meas'  : {0: [], 1: [], 2: [], 'all': []}, 
        'high_truth': {0: [], 1: [], 2: [], 'all': []}, 
        'high_pred' : {0: [], 1: [], 2: [], 'all': []},
        'high_pred_direct' : {0: [], 1: [], 2: [], 'all': []},
    }

    for comp_br in self.high_e_pred_comp.keys():
        _name = comp_br.replace('e_pred_raw_', '')
        dict_e_layer[comp_br] = {0: [], 1: [], 2: [], 'all': []}

    for ev_i in range(self.n_events):
        for layer in range(3):
            dict_e_layer['low_meas'][layer].append(
                (self.low_e_measured[ev_i][self.low_layer[ev_i] == layer]).sum())
            dict_e_layer['high_truth'][layer].append(
                (self.high_e_truth[ev_i][self.high_layer[ev_i] == layer]).sum())
            dict_e_layer['high_pred'][layer].append(
                (self.high_e_pred[ev_i][self.high_layer[ev_i] == layer]).sum())
            dict_e_layer['high_pred_direct'][layer].append(
                (self.high_e_pred_direct[ev_i][self.high_layer[ev_i] == layer]).sum())

            for comp_br in self.high_e_pred_comp.keys():
                dict_e_layer[comp_br][layer].append(
                    (self.high_e_pred_comp[comp_br][ev_i][self.high_layer[ev_i] == layer]).sum())

        dict_e_layer['low_meas']['all'].append(self.low_e_measured[ev_i].sum())
        dict_e_layer['high_truth']['all'].append(self.high_e_truth[ev_i].sum())
        dict_e_layer['high_pred']['all'].append(self.high_e_pred[ev_i].sum())
        dict_e_layer['high_pred_direct']['all'].append(self.high_e_pred_direct[ev_i].sum())

        for comp_br in self.high_e_pred_comp.keys():
            dict_e_layer[comp_br]['all'].append(self.high_e_pred_comp[comp_br][ev_i].sum())

    for k in dict_e_layer.keys():
        for layer in dict_e_layer[k].keys():
            dict_e_layer[k][layer] = np.array(dict_e_layer[k][layer])

    if truth_e_range is not None:
        for comp_br in self.high_e_pred_comp.keys():
            for layer in dict_e_layer[comp_br].keys():
                mask = (dict_e_layer['high_truth'][layer] > truth_e_range[0]) & (dict_e_layer['high_truth'][layer] < truth_e_range[1])
                dict_e_layer[comp_br][layer] = dict_e_layer[comp_br][layer][mask]

        for k in ['low_meas', 'high_pred', 'high_pred_direct', 'high_truth']: # the order is important, cause we are masking the same array with high_truth
            for layer in dict_e_layer[k].keys():
                mask = (dict_e_layer['high_truth'][layer] > truth_e_range[0]) & (dict_e_layer['high_truth'][layer] < truth_e_range[1])
                dict_e_layer[k][layer] = dict_e_layer[k][layer][mask]
        

    fig = plt.figure(figsize=(16,8), dpi=200)
    gs = GridSpec(2, 4, hspace=0.3, wspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    meas_truth = dict_e_layer['low_meas']['all'] - dict_e_layer['high_truth']['all']
    pred_truth = dict_e_layer['high_pred']['all'] - dict_e_layer['high_truth']['all']
    pred_direct_truth = dict_e_layer['high_pred_direct']['all'] - dict_e_layer['high_truth']['all']
    comb = np.hstack([meas_truth, pred_truth, pred_direct_truth])
    bins = np.linspace(np.percentile(comb, 1), np.percentile(comb, 99), 30)
    for comp_br in self.high_e_pred_comp.keys():
        comp_truth = dict_e_layer[comp_br]['all'] - dict_e_layer['high_truth']['all']
        ax1.hist(comp_truth, bins=bins, histtype='stepfilled', zorder=10, alpha=0.3)
    ax1.hist(meas_truth, bins=bins, histtype='stepfilled', color='cornflowerblue', lw=0.5, label='X = LR (meas)', alpha=0.8, zorder=10)
    ax1.hist(pred_truth, bins=bins, histtype='step', ec='r', lw=0.8, label='X = HR (pred)', zorder=10)
    ax1.hist(pred_direct_truth, bins=bins, histtype='step', ec='g', lw=0.8, label='X = HR (direct)', zorder=10)
    ax1.set_xlabel(r'$E_{X} -  E_{truth}$ [MeV]')

    ax2 = fig.add_subplot(gs[4])
    with np.errstate(divide='ignore', invalid='ignore'):
        r_meas_truth = meas_truth/dict_e_layer['high_truth']['all']
        r_pred_truth = pred_truth/dict_e_layer['high_truth']['all']
        r_pred_direct_truth = pred_direct_truth/dict_e_layer['high_truth']['all']
        r_meas_truth = r_meas_truth[np.isfinite(r_meas_truth)]; r_pred_truth = r_pred_truth[np.isfinite(r_pred_truth)]; r_pred_direct_truth = r_pred_direct_truth[np.isfinite(r_pred_direct_truth)]
        comb = np.hstack([r_meas_truth, r_pred_truth, r_pred_direct_truth])
        bins = np.linspace(np.percentile(comb, 1), np.percentile(comb, 99), 30)
        ax2.hist(r_meas_truth, bins=bins, histtype='stepfilled', 
            color='cornflowerblue', lw=0.5, label='X = LR (meas)', alpha=0.8, zorder=10)
        for comp_br in self.high_e_pred_comp.keys():
            comp_truth = dict_e_layer[comp_br]['all'] - dict_e_layer['high_truth']['all']
            ax2.hist(comp_truth/dict_e_layer['high_truth']['all'], bins=bins, histtype='stepfilled', zorder=10, alpha=0.3)
        ax2.hist(r_pred_truth, bins=bins, histtype='step', ec='r', lw=0.8, label='X = HR (pred)', zorder=10)
        ax2.hist(r_pred_direct_truth, bins=bins, histtype='step', ec='g', lw=0.8, label='X = HR (direct)', zorder=10)
    ax2.set_xlabel(r'$(E_{X} -  E_{truth}) / E_{truth}$')
    
    for ax in [ax1, ax2]:
        title = 'All layers'
        if truth_e_range is not None:
            title += f' ({truth_e_range[0]} < E < {truth_e_range[1]})'
        ax.set_title(title)
        ax.grid(True)

    for layer in range(3):
        ax = fig.add_subplot(gs[layer+1])
        meas_truth = dict_e_layer['low_meas'][layer] - dict_e_layer['high_truth'][layer]
        pred_truth = dict_e_layer['high_pred'][layer] - dict_e_layer['high_truth'][layer]
        pred_direct_truth = dict_e_layer['high_pred_direct'][layer] - dict_e_layer['high_truth'][layer]
        comb = np.hstack([meas_truth, pred_truth, pred_direct_truth])
        bins = np.linspace(np.percentile(comb, 1), np.percentile(comb, 99), 30)
        ax.hist(meas_truth, bins=bins, histtype='stepfilled', color='cornflowerblue', lw=0.5, label='X = LR (meas)', alpha=0.8, zorder=10)
        for comp_br in self.high_e_pred_comp.keys():
            comp_truth = dict_e_layer[comp_br][layer] - dict_e_layer['high_truth'][layer]
            ax.hist(comp_truth, bins=bins, histtype='stepfilled', zorder=10, alpha=0.3)
        ax.hist(pred_truth, bins=bins, histtype='step', ec='r', lw=0.8, label='X = HR (pred)', zorder=10)
        ax.hist(pred_direct_truth, bins=bins, histtype='step', ec='g', lw=0.8, label='X = HR (direct)', zorder=10)
        ax.set_xlabel(r'$E_{X} -  E_{truth}$ [MeV]')
        title = f'ECAL{layer+1}'
        if truth_e_range is not None:
            title += f' ({truth_e_range[0]} < E < {truth_e_range[1]})'
        ax.set_title(title)
        ax.grid(True)

        ax = fig.add_subplot(gs[4+layer+1])

        # get the max of the absolute value of the ratio while ignoring the Nans after division
        with np.errstate(divide='ignore', invalid='ignore'):
            r_meas_truth = meas_truth/dict_e_layer['high_truth'][layer]
            r_pred_truth = pred_truth/dict_e_layer['high_truth'][layer]
            r_pred_direct_truth = pred_direct_truth/dict_e_layer['high_truth'][layer]
        r_meas_truth = r_meas_truth[np.isfinite(r_meas_truth)]; r_pred_truth = r_pred_truth[np.isfinite(r_pred_truth)]; r_pred_direct_truth = r_pred_direct_truth[np.isfinite(r_pred_direct_truth)]
        comb = np.hstack([r_meas_truth, r_pred_truth, r_pred_direct_truth])
        bins = np.linspace(np.percentile(comb, 1), np.percentile(comb, 99), 30)
        ax.hist(r_meas_truth, bins=bins, histtype='stepfilled', color='cornflowerblue', lw=0.5, label='X = LR (meas)', alpha=0.8, zorder=10)
        for comp_br in self.high_e_pred_comp.keys():
            comp_truth = dict_e_layer[comp_br][layer] - dict_e_layer['high_truth'][layer]
            ax.hist(comp_truth/dict_e_layer['high_truth'][layer], bins=bins, histtype='stepfilled', zorder=10, alpha=0.3)
        ax.hist(r_pred_truth, bins=bins, histtype='step', color='cornflowerblue', ec='r', lw=0.5, label='HR (pred)', zorder=10)
        ax.hist(r_pred_direct_truth, bins=bins, histtype='step', color='cornflowerblue', ec='g', lw=5, label='HR (direct)', zorder=10)
        ax.set_xlabel(r'$(E_{X} -  E_{truth}) / E_{truth}$')
        ax.set_title(title)
        ax.grid(True)

    # plot legend at the top of the plot
    handles, labels = ax1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(0.0, 1.25), ncol=2, fancybox=True, shadow=True)

    # plt.legend(loc='upper right', ncol=2, fancybox=True, shadow=True) # bbox_to_anchor=(0.4, 0.8)

    if dir is None:
        meas_truth = dict_e_layer['low_meas']['all'] - dict_e_layer['high_truth']['all']
        pred_truth = dict_e_layer['high_pred']['all'] - dict_e_layer['high_truth']['all']
        ret_dict = {
            'res_meas_mean': meas_truth.mean(), 'res_meas_std': meas_truth.std(),
            'res_pred_mean': pred_truth.mean(), 'res_pred_std': pred_truth.std()
        }
        return fig, ret_dict

    plt.savefig(os.path.join(dir, 'residual_event_ensemble.png'))



def plot_ensemble_size_comparison(self, ens_avg_dict):
    fig = plt.figure(figsize=(10, 7), dpi=200)
    gs = GridSpec(2, 2, hspace=0.3, wspace=0.4, height_ratios=[1.5, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])


    # compute low resolution
    residuals_low = []; rel_residuals_low = []
    for i in range(self.n_events):
        res = self.low_e_measured[i].sum() - self.high_e_truth[i].sum()
        residuals_low.append(res)

        rel_res = res / self.high_e_truth[i].sum()
        rel_residuals_low.append(rel_res)

    residuals_low = np.array(residuals_low)
    rel_residuals_low = np.array(rel_residuals_low)

    q = 3
    max_abs_res = max(abs(np.percentile(residuals_low, q)), abs(np.percentile(residuals_low, 100-q)))
    max_abs_rel_res = max(abs(np.percentile(rel_residuals_low, q)), abs(np.percentile(rel_residuals_low, 100-q)))



    individual_residuals = {}
    individual_rel_residuals = {}

    # compute all the individual ensemble components
    for k, v in self.high_e_pred_raw_comp.items():
        residuals_comp = []; rel_residuals_comp = []
        for i in range(self.n_events):
            res = v[i].sum() - self.high_e_truth[i].sum()
            residuals_comp.append(res)

            rel_res = res / self.high_e_truth[i].sum()
            rel_residuals_comp.append(rel_res)

        individual_residuals[k] = np.array(residuals_comp)
        individual_rel_residuals[k] = np.array(rel_residuals_comp)
    
    # min and max of the residuals
    for k, v in individual_residuals.items():
        max_abs_res = max(max_abs_res, abs(np.percentile(v, q)), abs(np.percentile(v, 100-q)))
        max_abs_rel_res = max(max_abs_rel_res, 
            abs(np.percentile(individual_rel_residuals[k], q)), 
            abs(np.percentile(individual_rel_residuals[k], 100-q)))

    bins_res = np.linspace(-max_abs_res, max_abs_res, 30)
    bins_rel_res = np.linspace(-max_abs_rel_res, max_abs_rel_res, 30)


    # plot the low resolution
    ax1.hist(residuals_low, bins=bins_res, histtype='stepfilled', alpha=0.8, label='low resolution', color='cornflowerblue')
    ax2.hist(rel_residuals_low, bins=bins_rel_res, histtype='stepfilled', alpha=0.8, label='low resolution', color='cornflowerblue')
    
    _min, _max = 0, max(ens_avg_dict.keys())
    res_mean, res_std = residuals_low.mean(), residuals_low.std()
    rel_res_mean, rel_res_std = rel_residuals_low.mean(), rel_residuals_low.std()

    # ax3.hlines(res_mean, _min, _max, color='cornflowerblue', label='low resolution')
    # ax3.fill_between([_min, _max], res_mean - res_std, res_mean + res_std, color='cornflowerblue', alpha=0.3)
    # ax4.hlines(rel_res_mean, _min, _max, color='cornflowerblue', label='low resolution')
    # ax4.fill_between([_min, _max], rel_res_mean - rel_res_std, rel_res_mean + rel_res_std, color='cornflowerblue', alpha=0.3)

    # plot the individual ensemble components
    for i, k in enumerate(individual_residuals.keys()):
        label = 'ensemble components' if i==0 else None
        ax1.hist(individual_residuals[k], bins=bins_res, histtype='step', alpha=0.3, label=label, color='k')
        ax2.hist(individual_rel_residuals[k], bins=bins_rel_res, histtype='step', alpha=0.3, label=label, color='k')

        # if i==0:
        #     ax3.errorbar([1], np.mean(individual_residuals[k]), yerr=np.std(individual_residuals[k]), fmt='o', color='k', label='ensemble components')
        #     ax4.errorbar([1], np.mean(individual_rel_residuals[k]), yerr=np.std(individual_rel_residuals[k]), fmt='o', color='k', label='ensemble components')
            

    # summary for paper
    x_vals = []; y_vals_res = []; y_vals_rel_res = []

    # plot the ensemble averages
    colors = plt.cm.viridis(np.linspace(0, 1, len(ens_avg_dict.keys()))) 
    for ens_i, (k, v) in enumerate(ens_avg_dict.items()):
        res = []; rel_res = []
        for ev_i in range(self.n_events):
            res_ev = v[ev_i].sum() - self.high_e_truth[ev_i].sum()
            rel_res_ev = res_ev / self.high_e_truth[ev_i].sum()
            res.append(res_ev)
            rel_res.append(rel_res_ev)
        ax1.hist(res, bins=bins_res, histtype='step', alpha=0.8, label=f'# components = {k}', color=colors[ens_i])
        ax2.hist(rel_res, bins=bins_rel_res, histtype='step', alpha=0.8, label=f'# components = {k}', color=colors[ens_i])
        
        # plot error bars
        # ax3.errorbar([k], np.mean(res), yerr=np.std(res), fmt='o', color=colors[ens_i])
        # ax4.errorbar([k], np.mean(rel_res), yerr=np.std(rel_res), fmt='o', color=colors[ens_i])

        x_vals.append(k)
        y_vals_res.append(np.std(res))
        y_vals_rel_res.append(np.std(rel_res))


    ax3.plot(x_vals, y_vals_res, color='k', marker='o')
    ax4.plot(x_vals, y_vals_rel_res, color='k', marker='o')


    # cosmetics
    ax1.set_xlabel(r'$E_{X} -  E_{truth}$ [MeV]')
    ax2.set_xlabel(r'$(E_{X} -  E_{truth}) / E_{truth}$')

    ax3.set_xlabel('#num ensemble components')
    ax4.set_xlabel('#num ensemble components')
    ax3.set_ylabel('width of the residuals'); ax4.set_ylabel('width of the relative residuals')

    ax1.set_title('Residuals'); ax3.set_title('Residuals')
    ax2.set_title('Relative residuals'); ax4.set_title('Relative residuals')

    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(True)

    # plot the legend above ax1
    ax1.legend(loc='upper left', bbox_to_anchor=(-0.05, 1.30), ncol=4, fancybox=True, shadow=True)

    return fig

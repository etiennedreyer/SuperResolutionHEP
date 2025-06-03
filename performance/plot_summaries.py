import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

from collections import OrderedDict
import os
from .util import get_mean_std_iqr_label



def plot_residual_event(self, dir=None, truth_e_range=None):

    dict_e_layer = {
        'low_meas'  : {0: [], 1: [], 2: [], 'all': []}, 
        'high_truth': {0: [], 1: [], 2: [], 'all': []}, 
        'high_pred' : {0: [], 1: [], 2: [], 'all': []}
    }
    for ev_i in range(self.n_events):
        for layer in range(3):
            dict_e_layer['low_meas'][layer].append(
                (self.low_e_measured[ev_i][self.low_layer[ev_i] == layer]).sum())
            dict_e_layer['high_truth'][layer].append(
                (self.high_e_truth[ev_i][self.high_layer[ev_i] == layer]).sum())
            dict_e_layer['high_pred'][layer].append(
                (self.high_e_pred[ev_i][self.high_layer[ev_i] == layer]).sum())

        dict_e_layer['low_meas']['all'].append(self.low_e_measured[ev_i].sum())
        dict_e_layer['high_truth']['all'].append(self.high_e_truth[ev_i].sum())
        dict_e_layer['high_pred']['all'].append(self.high_e_pred[ev_i].sum())
    
    for k in dict_e_layer.keys():
        for layer in dict_e_layer[k].keys():
            dict_e_layer[k][layer] = np.array(dict_e_layer[k][layer])

    if truth_e_range is not None:
        for k in ['low_meas', 'high_pred', 'high_truth']: # the order is important, cause we are masking the same array with high_truth
            for layer in dict_e_layer[k].keys():
                mask = (dict_e_layer['high_truth'][layer] > truth_e_range[0]) & (dict_e_layer['high_truth'][layer] < truth_e_range[1])
                dict_e_layer[k][layer] = dict_e_layer[k][layer][mask]

    fig = plt.figure(figsize=(16,8), dpi=200)
    gs = GridSpec(2, 4, hspace=0.8, wspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    meas_truth = dict_e_layer['low_meas']['all'] - dict_e_layer['high_truth']['all']
    pred_truth = dict_e_layer['high_pred']['all'] - dict_e_layer['high_truth']['all']
    comb = np.hstack([meas_truth, pred_truth])
    bins = np.linspace(np.percentile(comb, 1), np.percentile(comb, 99), 30)
    label, _ = get_mean_std_iqr_label(meas_truth, precision=1)
    ax1.hist(meas_truth, bins=bins, histtype='stepfilled', color='cornflowerblue', lw=0.5, label=label, alpha=0.8, zorder=10)
    label, _ = get_mean_std_iqr_label(pred_truth, precision=1)
    ax1.hist(pred_truth, bins=bins, histtype='step', ec='r', lw=0.8, label=label, zorder=10)
    ax1.set_xlabel(r'$E_{X} -  E_{truth}$ [MeV]')

    # plot legend at the top of the plot
    handles, labels = ax1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(0.0, 1.25), ncol=1, fancybox=False, shadow=False)

    ax2 = fig.add_subplot(gs[4])
    with np.errstate(divide='ignore', invalid='ignore'):
        r_meas_truth = meas_truth/dict_e_layer['high_truth']['all']
        r_pred_truth = pred_truth/dict_e_layer['high_truth']['all']
        r_meas_truth = r_meas_truth[np.isfinite(r_meas_truth)]; r_pred_truth = r_pred_truth[np.isfinite(r_pred_truth)]
        comb = np.hstack([r_meas_truth, r_pred_truth])
        bins = np.linspace(np.percentile(comb, 1), np.percentile(comb, 99), 30)
        label, _ = get_mean_std_iqr_label(r_meas_truth)
        ax2.hist(r_meas_truth, bins=bins, histtype='stepfilled', 
            color='cornflowerblue', lw=0.5, label=label, alpha=0.8, zorder=10)
        label, _ = get_mean_std_iqr_label(r_pred_truth)
        ax2.hist(r_pred_truth, bins=bins, histtype='step', ec='r', lw=0.8, label=label, zorder=10)
    ax2.set_xlabel(r'$(E_{X} -  E_{truth}) / E_{truth}$')
    
    # plot legend at the top of the plot
    handles, labels = ax2.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(0.0, 1.25), ncol=1, fancybox=False, shadow=False)


    for ax in [ax1, ax2]:
        title = 'All layers'
        if truth_e_range is not None:
            title += f' ({truth_e_range[0]} < E < {truth_e_range[1]})'
        ax.set_title(title)
        ax.grid(True)

    for layer in range(3):
        if dict_e_layer['low_meas'][layer].shape[0] == 0:
            continue
        ax = fig.add_subplot(gs[layer+1])
        meas_truth = dict_e_layer['low_meas'][layer] - dict_e_layer['high_truth'][layer]
        pred_truth = dict_e_layer['high_pred'][layer] - dict_e_layer['high_truth'][layer]
        comb = np.hstack([meas_truth, pred_truth])
        bins = np.linspace(np.percentile(comb, 1), np.percentile(comb, 99), 30)
        ax.hist(meas_truth, bins=bins, histtype='stepfilled', color='cornflowerblue', lw=0.5, label='X = LR (meas)', alpha=0.8, zorder=10)
        ax.hist(pred_truth, bins=bins, histtype='step', ec='r', lw=0.8, label='X = HR (pred)', zorder=10)
        ax.set_xlabel(r'$E_{X} -  E_{truth}$ [MeV]')
        title = f'ECAL{layer+1}'
        if truth_e_range is not None:
            title += f' ({truth_e_range[0]} < E < {truth_e_range[1]})'
        ax.set_title(title)
        ax.grid(True)

        if layer == 2:
            # plot legend at the top of the plot
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(-0.4, 1.35), ncol=2, fancybox=True, shadow=True)


        ax = fig.add_subplot(gs[4+layer+1])

        # get the max of the absolute value of the ratio while ignoring the Nans after division
        with np.errstate(divide='ignore', invalid='ignore'):
            r_meas_truth = meas_truth/dict_e_layer['high_truth'][layer]
            r_pred_truth = pred_truth/dict_e_layer['high_truth'][layer]
        r_meas_truth = r_meas_truth[np.isfinite(r_meas_truth)]; r_pred_truth = r_pred_truth[np.isfinite(r_pred_truth)]
        comb = np.hstack([r_meas_truth, r_pred_truth])
        if len(comb) == 0:
            continue
        bins = np.linspace(np.percentile(comb, 1), np.percentile(comb, 99), 30)
        ax.hist(r_meas_truth, bins=bins, histtype='stepfilled', color='cornflowerblue', lw=0.5, label='X = LR (meas)', alpha=0.8, zorder=10)
        ax.hist(r_pred_truth, bins=bins, histtype='step', color='cornflowerblue', ec='r', lw=0.5, label='HR (pred)', zorder=10)
        ax.set_xlabel(r'$(E_{X} -  E_{truth}) / E_{truth}$')
        ax.set_title(title)
        ax.grid(True)


    # plt.legend(loc='upper right', ncol=2, fancybox=True, shadow=True) # bbox_to_anchor=(0.4, 0.8)

    if dir is None:
        meas_truth = dict_e_layer['low_meas']['all'] - dict_e_layer['high_truth']['all']
        pred_truth = dict_e_layer['high_pred']['all'] - dict_e_layer['high_truth']['all']
        ret_dict = {
            'res_meas_mean': meas_truth.mean(), 'res_meas_std': meas_truth.std(),
            'res_pred_mean': pred_truth.mean(), 'res_pred_std': pred_truth.std()
        }
        return fig, ret_dict

    plt.savefig(os.path.join(dir, 'residual_event.png'))




def plot_residual_cell(self, dir=None):

    e_truth = np.hstack(self.high_e_truth)
    e_pred  = np.hstack(self.high_e_pred) 
    e_diff  = e_pred - e_truth
    
    fig = plt.figure(figsize=(16, 4), dpi=200)
    gs = GridSpec(1, 4, hspace=0.3, wspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    iqr = np.subtract(*np.percentile(e_diff, [75, 25]))
    bins = np.linspace(-1*iqr, 1*iqr, 30)
    ax1.hist(e_diff, bins=bins, histtype='stepfilled', color='cornflowerblue', ec='k', lw=0.5, alpha=0.8, zorder=10)
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$E_{pred} -  E_{truth}$ [MeV]')
    ax1.set_title('All cells')


    ax2 = fig.add_subplot(gs[1])
    ratio_e_diff = e_diff/(e_truth+1e-8)
    bins = np.linspace(np.percentile(ratio_e_diff, 1), np.percentile(ratio_e_diff, 99), 30)
    ax2.hist(e_diff/(e_truth+1e-8), bins=bins, histtype='stepfilled', color='cornflowerblue', ec='k', lw=0.5, alpha=0.8, zorder=10)
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$(E_{pred} -  E_{truth})/E_{truth}$')
    ax2.set_title('All cells')

    ax3 = fig.add_subplot(gs[2])
    ax3.hist2d(e_truth, e_diff, bins=30, norm=mpl.colors.LogNorm(), cmap='plasma_r', zorder=10)
    ax3.set_xlabel(r'$E_{truth}$ [MeV]')
    ax3.set_ylabel(r'$E_{pred} -  E_{truth}$ [MeV]')
    ax3.set_title('All cells')

    ax4 = fig.add_subplot(gs[3])
    bins = np.linspace(e_truth.min(), e_truth.max()+1, 25)
    ys, y_errs = [], []
    for i in range(len(bins)-1):
        mask = (e_truth >= bins[i]) & (e_truth < bins[i+1])
        if mask.sum() == 0:
            ys.append(np.nan); y_errs.append(np.nan)
            continue
        ys.append(e_diff[mask].mean())
        y_errs.append(e_diff[mask].std())
    xs = (bins[1:] + bins[:-1])*0.5
    ax4.errorbar(xs, ys, yerr=y_errs, fmt='o', color='cornflowerblue', ecolor='k', lw=0.5, zorder=10)
    ax4.set_xlabel(r'$E_{truth}$ [MeV]')
    ax4.set_ylabel(r'$E_{pred} -  E_{truth}$ [MeV]')
    ax4.set_title('All cells')

    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(True)

    if dir is None:
        return fig
    
    plt.savefig(os.path.join(dir, 'residual_cell.png'))



def plot_residual_cell_for_one_event(self, event_idx, dir=None):

    e_truth = self.high_e_truth[event_idx]
    e_pred  = self.high_e_pred[event_idx]
    e_diff  = e_pred - e_truth

    fig = plt.figure(figsize=(4,4), dpi=200)
    gs = GridSpec(1, 1, hspace=0.3, wspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(e_truth, e_diff, s=1)
    ax1.set_xlabel(r'$E_{truth}$')
    ax1.set_ylabel(r'$E_{pred} -  E_{truth}$')
    ax1.set_title('All cells')

    for ax in [ax1]:
        ax.set_axisbelow(True)
        ax.grid()

    if dir is None:
        return fig
    
    plt.savefig(os.path.join(dir, f'residual_cell_event_{event_idx}.png'))
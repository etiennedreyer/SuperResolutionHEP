import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib as mpl

import os


def plot_evolution(self, idx, check_binning=False, dir=None):

    def e_trans(e):
        e = np.clip(e, 0, None)
        return np.log(e+1)

    def get_text(x):
        text = f'sum = {x.sum():.0f} MeV \npeak = {x.max():.0f} MeV'
        return text

    cmap = mpl.cm.get_cmap("plasma_r").copy()
    cmap.set_under(color='white')
    title_fontsize = 18

    low_eta    = self.low_eta[idx]
    low_phi    = self.low_phi[idx]
    low_layer  = self.low_layer[idx]
    low_e_meas = self.low_e_measured[idx]

    high_eta    = self.high_eta[idx]
    high_phi    = self.high_phi[idx]
    high_layer  = self.high_layer[idx]
    high_e_pred  = self.high_e_pred[idx]
    high_e_truth = self.high_e_truth[idx]
    high_e_proxy = self.high_e_proxy[idx]

    high_e_pred_step = {}
    for k, v in self.high_e_pred_step.items():
        high_e_pred_step[k] = v[idx]

    vmin = min(high_e_truth.min(), high_e_pred.min(), low_e_meas.min())
    vmax = max(high_e_truth.max(), high_e_pred.max(), low_e_meas.max())
    for k in self.high_e_pred_step.keys():
        vmin = min(vmin, high_e_pred_step[k].min())
        vmax = max(vmax, high_e_pred_step[k].max())

    vmin, vmax = 1, e_trans(vmax)

    xmin, xmax = \
        min(low_eta.min(), high_eta.min()), max(low_eta.max(), high_eta.max())
    ymin, ymax = \
        min(low_phi.min(), high_phi.min()), max(low_phi.max(), high_phi.max())
        
    xrange, yrange = xmax - xmin, ymax - ymin

    xmin, xmax = xmin - xrange/1000, xmax + xrange/1000
    ymin, ymax = ymin - yrange/1000, ymax + yrange/1000

    xgrid_, ygrid_ = np.linspace(-3,3,64), np.linspace(-np.pi,np.pi,64)
    
    xmin = xgrid_[np.abs(xgrid_ - (xmin - (xgrid_[1]-xgrid_[0]))).argmin()]
    xmax = xgrid_[np.abs(xgrid_ - (xmax + (xgrid_[1]-xgrid_[0]))).argmin()]
    ymin = ygrid_[np.abs(ygrid_ - (ymin - (ygrid_[1]-ygrid_[0]))).argmin()]
    ymax = ygrid_[np.abs(ygrid_ - (ymax + (ygrid_[1]-ygrid_[0]))).argmin()]

    ncol = len(self.high_e_pred_step.keys()) + 3
    fig = plt.figure(figsize=(3*ncol, 3*3.2), dpi=300) # dpi=200
    gs_outer = GridSpec(3, 2, hspace=0.5, wspace=0.1, width_ratios=[3, ncol-3.6])

    for i in range(3):
        mask_low  = low_layer  == i
        mask_high = high_layer == i

        # add gridspec inside the outer gridspec
        gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[i, 0], wspace=0.38)
        ax1 = fig.add_subplot(gs[0])

        _, _, _, im1 = ax1.hist2d(
            low_eta[mask_low], low_phi[mask_low], weights=e_trans(low_e_meas[mask_low]), 
            bins=[np.linspace(-3,3,self.low_gran[i]+1), np.linspace(-np.pi,np.pi,self.low_gran[i]+1)],
            cmap=cmap, vmin=vmin, vmax=vmax)
        # ax1.text(0.05, 0.8, get_text(low_e_meas[mask_low]), transform=ax1.transAxes)
        ax1.set_title("LR (measured)", fontsize=title_fontsize)
        ax1.annotate(f'ECAL{i+1}', xy=(0, 0.5), xytext=(-ax1.yaxis.labelpad - 5, 0),
            xycoords=ax1.yaxis.label, textcoords='offset points',
            ha='right', va='center', fontsize=22, rotation=90)


        if i == 0:
            cax = fig.add_axes([0.04, 0.11, 0.005, 0.77])
            cbar = fig.colorbar(im1, cax=cax)
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.ax.tick_params(labelsize=12)
            
            # Set the label at the top and rotate it
            cbar.set_label('$ln \\left( E + 1 \\right)$', labelpad=5, fontsize=18)
            cbar.ax.yaxis.set_label_position('left')
            # cbar.ax.yaxis.label.set_rotation(0)
            cbar.ax.yaxis.label.set_horizontalalignment('center')
            cbar.ax.yaxis.label.set_verticalalignment('bottom')        

        ax2 = fig.add_subplot(gs[1])
        ax2.hist2d(
            high_eta[mask_high], high_phi[mask_high], weights=e_trans(high_e_truth[mask_high]), 
            bins=[np.linspace(-3,3,self.high_gran[i]+1), np.linspace(-np.pi,np.pi,self.high_gran[i]+1)], 
            cmap=cmap, vmin=vmin, vmax=vmax)
        # ax2.text(0.05, 0.8, get_text(high_e_truth[mask_high]), transform=ax2.transAxes)
        ax2.set_title(f"HR (truth)", fontsize=title_fontsize)

        ax3 = fig.add_subplot(gs[2])
        h, x, _, _ = ax3.hist2d(
            high_eta[mask_high], high_phi[mask_high], weights=e_trans(high_e_pred[mask_high]), 
            bins=[np.linspace(-3,3,self.high_gran[i]+1), np.linspace(-np.pi,np.pi,self.high_gran[i]+1)],
            cmap=cmap, vmin=vmin, vmax=vmax)
        # ax3.text(0.05, 0.8, get_text(high_e_pred[mask_high]), transform=ax3.transAxes)
        ax3.set_title(f"HR (predicted) t = 1", fontsize=title_fontsize)

        for ax in [ax1, ax2, ax3]:
            ax.set_xlim([xmin, xmax]); ax.set_ylim([ymin, ymax])
            ax.set_xlabel(r'$\eta$'); ax.set_ylabel(r'$\phi$')
            # ax.set_aspect('equal', adjustable='box')


        gs = GridSpecFromSubplotSpec(1, ncol-3, subplot_spec=gs_outer[i, 1], wspace=0.1)

        sorted_keys = list(self.high_e_pred_step.keys())
        sorted_keys.sort(key=lambda x: x.split('_')[-1])
        sorted_keys.reverse()

        for j, key in enumerate(sorted_keys):
            ax = fig.add_subplot(gs[j])
            ax.hist2d(
                high_eta[mask_high], high_phi[mask_high], weights=e_trans(high_e_pred_step[key][mask_high]), 
                bins=[np.linspace(-3,3,self.high_gran[i]+1), np.linspace(-np.pi,np.pi,self.high_gran[i]+1)],
                cmap=cmap, vmin=vmin, vmax=vmax)
            # ax.text(0.05, 0.8, get_text(high_e_pred_step[key][mask_high]), transform=ax.transAxes)
            ax.set_xlim([xmin, xmax]); ax.set_ylim([ymin, ymax])
            ax.set_title(f"t = {key.split('_')[-1]}", fontsize=title_fontsize)
            ax.set_xticks([]); ax.set_yticks([])
            # make it square
            # ax.set_aspect('equal', adjustable='box')


        # binning check; switch it off for speed up
        if check_binning:
            h1, _, _ = np.histogram2d(
                low_eta[mask_low], low_phi[mask_low], 
                bins=[np.linspace(-3,3,self.low_gran[i]+1), np.linspace(-np.pi,np.pi,self.low_gran[i]+1)])

            h2, _, _ = np.histogram2d(
                high_eta[mask_high], high_phi[mask_high],
                bins=[np.linspace(-3,3,self.high_gran[i]+1), np.linspace(-np.pi,np.pi,self.high_gran[i]+1)])
            
            if ((h1>1).sum() != 0) or ((h2>1).sum() != 0):
                print()
                print('Warning! Issue with binning. Found bins with >1 hits')
                print((h1>1).sum(), (h2>1).sum())

    if dir is None:
        return fig
    
    plt.savefig(os.path.join(dir, f'ED_{idx}.png'))


def plot_evolution_raw_nn(self, idx, dir=None, trans=None):

    if trans is None:
        def trans(x):
            return x
        
    def get_text(x):
        text = f'min = {x.min():.2f} \nmax = {x.max():.2f}'
        return text

    cmap = mpl.cm.get_cmap("plasma_r").copy()
    cmap.set_under(color='white')

    high_eta    = self.high_eta[idx]
    high_phi    = self.high_phi[idx]
    high_layer  = self.high_layer[idx]

    high_raw_nn_cond   = self.high_raw_nn_cond[idx]
    high_raw_nn_target = self.high_raw_nn_target[idx]
    high_raw_nn_pred   = self.high_raw_nn_pred[idx]

    high_raw_nn_pred_step = {}
    for k, v in self.high_raw_nn_pred_step.items():
        high_raw_nn_pred_step[k] = v[idx]

    vmin = min(high_raw_nn_target.min(), high_raw_nn_pred.min(), high_raw_nn_cond.min())
    vmax = max(high_raw_nn_target.max(), high_raw_nn_pred.max(), high_raw_nn_cond.max())

    for k in self.high_raw_nn_pred_step.keys():
        vmin = min(vmin, high_raw_nn_pred_step[k].min())
        vmax = max(vmax, high_raw_nn_pred_step[k].max())

    xmin, xmax = high_eta.min(), high_eta.max()
    ymin, ymax = high_phi.min(), high_phi.max()
        
    xrange, yrange = xmax - xmin, ymax - ymin
        
    xmin, xmax = xmin - xrange/5, xmax + xrange/5
    ymin, ymax = ymin - yrange/5, ymax + yrange/5

    xgrid_, ygrid_ = np.linspace(-3,3,64), np.linspace(-np.pi,np.pi,64)
    
    xmin = xgrid_[np.abs(xgrid_ - (xmin - (xgrid_[1]-xgrid_[0]))).argmin()]
    xmax = xgrid_[np.abs(xgrid_ - (xmax + (xgrid_[1]-xgrid_[0]))).argmin()]
    ymin = ygrid_[np.abs(ygrid_ - (ymin - (ygrid_[1]-ygrid_[0]))).argmin()]
    ymax = ygrid_[np.abs(ygrid_ - (ymax + (ygrid_[1]-ygrid_[0]))).argmin()]

    ncol = len(self.high_raw_nn_pred_step.keys()) + 3
    fig = plt.figure(figsize=(3*ncol, 3*3), dpi=200)
    gs = GridSpec(3, ncol, hspace=0.3, wspace=0.2)

    for i in range(3):
        mask_high = high_layer == i
        bins = [np.linspace(-3,3,self.high_gran[i]+1), np.linspace(-np.pi,np.pi,self.high_gran[i]+1)]

        bin_x_indices = np.digitize(high_eta[mask_high], bins[0]) - 1
        bin_y_indices = np.digitize(high_phi[mask_high], bins[1]) - 1

        bin_centers_eta = (bins[0][:-1] + bins[0][1:]) / 2
        bin_centers_phi = (bins[1][:-1] + bins[1][1:]) / 2
        bin_centers_eta, bin_centers_phi = np.meshgrid(bin_centers_eta, bin_centers_phi)
        bin_centers_eta, bin_centers_phi = bin_centers_eta.T.flatten(), bin_centers_phi.T.flatten()


        weight_matrix = np.zeros((len(bins[0]) - 1, len(bins[1]) - 1)) - 1000
                            
        ax1 = fig.add_subplot(gs[i*ncol])
        weight_matrix[bin_x_indices, bin_y_indices] = trans(high_raw_nn_cond[mask_high])
        _, _, _, im1 = ax1.hist2d(
            bin_centers_eta, bin_centers_phi, weights=weight_matrix.flatten(),
            bins=bins, vmin=vmin, vmax=vmax, cmap=cmap)
        ax1.text(0.05, 0.8, get_text(trans(high_raw_nn_cond[mask_high])), transform=ax1.transAxes)
        ax1.set_title(f"raw NN cond")

        if i == 0:
            cax = fig.add_axes([0.09, 0.2, 0.005, 0.6])
            cbar = fig.colorbar(im1, cax=cax)
            cbar.ax.yaxis.set_ticks_position('left')

        ax2 = fig.add_subplot(gs[i*ncol + 1])
        weight_matrix[bin_x_indices, bin_y_indices] = trans(high_raw_nn_target[mask_high])
        ax2.hist2d(
            bin_centers_eta, bin_centers_phi, weights=weight_matrix.flatten(),
            bins=bins, vmin=vmin, vmax=vmax, cmap=cmap)
        ax2.text(0.05, 0.8, get_text(trans(high_raw_nn_target[mask_high])), transform=ax2.transAxes)
        ax2.set_title(f"raw NN target")

        ax3 = fig.add_subplot(gs[i*ncol + 2])
        weight_matrix[bin_x_indices, bin_y_indices] = trans(high_raw_nn_pred[mask_high])
        ax3.hist2d(
            bin_centers_eta, bin_centers_phi, weights=weight_matrix.flatten(),
            bins=bins, vmin=vmin, vmax=vmax, cmap=cmap)
        ax3.text(0.05, 0.8, get_text(trans(high_raw_nn_pred[mask_high])), transform=ax3.transAxes)
        ax3.set_title(f"raw NN pred")

        for ax in [ax1, ax2, ax3]:
            ax.set_xlim([xmin, xmax]); ax.set_ylim([ymin, ymax])
                
        sorted_keys = list(self.high_raw_nn_pred_step.keys())
        sorted_keys.sort(key=lambda x: x.split('_')[-1])
        sorted_keys.reverse()

        for j, key in enumerate(sorted_keys):
            ax = fig.add_subplot(gs[i*ncol + 3+j])
            weight_matrix[bin_x_indices, bin_y_indices] = trans(high_raw_nn_pred_step[key][mask_high])
            ax.hist2d(
                bin_centers_eta, bin_centers_phi, weights=weight_matrix.flatten(),
                bins=bins, vmin=vmin, vmax=vmax, cmap=cmap)
            ax.text(0.05, 0.8, get_text(high_raw_nn_pred_step[key][mask_high]), transform=ax.transAxes)
            ax.set_xlim([xmin, xmax]); ax.set_ylim([ymin, ymax])
            ax.set_title(f"t = {key.split('_')[-1]}")

    if dir is None:
        return fig
    
    plt.savefig(os.path.join(dir, f'raw_NN_ED_{idx}.png'))


def plot_evolution_raw_nn_dist(self, idx, dir=None, trans=None):

    if trans is None:
        def trans(x):
            return x
        
    def get_text(x):
        text = f'min = {x.min():.2f} \nmax = {x.max():.2f}'
        return text

    high_layer  = self.high_layer[idx]

    high_raw_nn_cond   = self.high_raw_nn_cond[idx]
    high_raw_nn_target = self.high_raw_nn_target[idx]
    high_raw_nn_pred   = self.high_raw_nn_pred[idx]

    high_raw_nn_pred_step = {}
    for k, v in self.high_raw_nn_pred_step.items():
        high_raw_nn_pred_step[k] = v[idx]

    ncol = len(self.high_raw_nn_pred_step.keys()) + 3
    fig = plt.figure(figsize=(3*ncol, 3*3), dpi=200)
    gs = GridSpec(3, ncol, hspace=0.3, wspace=0.2)

    for i in range(3):
        mask_high = high_layer == i

        vmin = min(high_raw_nn_target[mask_high].min(), high_raw_nn_pred[mask_high].min(), high_raw_nn_cond[mask_high].min())
        vmax = max(high_raw_nn_target[mask_high].max(), high_raw_nn_pred[mask_high].max(), high_raw_nn_cond[mask_high].max())

        for k in self.high_raw_nn_pred_step.keys():
            vmin = min(vmin, high_raw_nn_pred_step[k][mask_high].min())
            vmax = max(vmax, high_raw_nn_pred_step[k][mask_high].max())

        bins = np.linspace(vmin, vmax, 30)

        ax1 = fig.add_subplot(gs[i*ncol])
        vals = trans(high_raw_nn_cond[mask_high])
        ax1.hist(vals, bins=bins, histtype='stepfilled', color='cornflowerblue', lw=0.5, ec='k')
        ax1.text(0.5, 0.8, get_text(trans(high_raw_nn_cond[mask_high])), transform=ax1.transAxes)
        ax1.set_title(f"raw NN cond")

        ax2 = fig.add_subplot(gs[i*ncol + 1])
        vals = trans(high_raw_nn_target[mask_high])
        ax2.hist(vals, bins=bins, histtype='stepfilled', color='cornflowerblue', lw=0.5, ec='k')
        ax2.text(0.5, 0.8, get_text(trans(high_raw_nn_target[mask_high])), transform=ax2.transAxes)
        ax2.set_title(f"raw NN target")

        ax3 = fig.add_subplot(gs[i*ncol + 2])
        vals = trans(high_raw_nn_pred[mask_high])
        ax3.hist(vals, bins=bins, histtype='stepfilled', color='cornflowerblue', lw=0.5, ec='k')
        ax3.text(0.5, 0.8, get_text(trans(high_raw_nn_pred[mask_high])), transform=ax3.transAxes)
        ax3.set_title(f"raw NN pred")
                
        vals = trans(high_raw_nn_target[mask_high])
        ax3.hist(vals, bins=bins, histtype='step', color='red', lw=1)

        sorted_keys = list(self.high_raw_nn_pred_step.keys())
        sorted_keys.sort(key=lambda x: x.split('_')[-1])
        sorted_keys.reverse()

        for j, key in enumerate(sorted_keys):
            ax = fig.add_subplot(gs[i*ncol + 3+j])
            vals = trans(high_raw_nn_pred_step[key][mask_high])
            ax.hist(vals, bins=bins, histtype='stepfilled', color='cornflowerblue', lw=0.5, ec='k')
            ax.text(0.5, 0.8, get_text(trans(high_raw_nn_pred_step[key][mask_high])), transform=ax.transAxes)
            ax.set_title(f"t = {key.split('_')[-1]}")

    if dir is None:
        return fig
    
    plt.savefig(os.path.join(dir, f'raw_NN_dist_ED_{idx}.png'))
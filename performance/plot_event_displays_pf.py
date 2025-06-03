import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib as mpl
from matplotlib.lines import Line2D

import os



def get_rgb(self, res, idx, argmax=False):
    energy_incidence = []
    for i in range(self.max_part):
        if res == 'lr':
            energy_incidence.append(self.inc_wt_lr_pf[i][idx])
        elif res == 'hr':
            energy_incidence.append(self.inc_wt_hr_pf[i][idx])
    energy_incidence = np.array(energy_incidence).T

    weights = energy_incidence
    if argmax:
        weights = np.zeros_like(energy_incidence)
        weights[np.arange(energy_incidence.shape[0]), np.argmax(energy_incidence, axis=1)] = 1

    rgb = np.dot(weights, self.pf_colors)

    return rgb



def plot_pf_event_display(self, idx, dir=None, verbose=False):

    def e_trans(e):
        e = np.clip(e, 0, None)
        return np.log(e+1)

    def get_text(x):
        text = ''
        if len(x) > 0:
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

    if (low_e_meas > 1).sum() != self.inc_wt_lr_pf[0][idx].shape[0]:
        print(f'Low res measured energy shape (', low_e_meas.shape, ') != low res PF energy shape (', self.inc_wt_lr_pf[0][idx].shape, '). Skipping this event.')
        return

    if (high_e_pred > 1).sum() != self.inc_wt_hr_pf[0][idx].shape[0]:
        print((high_e_pred > 1).sum(), self.inc_wt_hr_pf[0][idx].shape[0])
        print('High res truth energy shape (', high_e_truth.shape, ') != high res PF energy shape (', self.inc_wt_hr_pf[0][idx].shape, '). Skipping this event.')
        return

    attn_rgbs_high = self.get_rgb('hr', idx)
    attn_rgbs_low = self.get_rgb('lr', idx)

    vmin = min(high_e_truth.min(), high_e_pred.min(), low_e_meas.min())
    vmax = max(high_e_truth.max(), high_e_pred.max(), low_e_meas.max())
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

    fig = plt.figure(figsize=(21, 13), dpi=300)
    gs_outer = GridSpec(3, 3, hspace=0.43, wspace=0.3, width_ratios=[1, 1, 0.1])
    # fig.suptitle(f'Event {idx}; truth card {len(self.truth_part_dep_e[idx])}', fontsize=16)

    # legend
    gs_legend = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[:, 2], height_ratios=[1, 2])
    ax_legend = fig.add_subplot(gs_legend[0])
    ax_legend.axis('off')  # Hide the axes

    colors = [self.pf_colors[0], self.pf_colors[1], self.pf_colors[2]]
    labels = ['Particle 1', 'Particle 2', 'Particle 3']

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=30, label=label) for color, label in zip(colors, labels)]
    ax_legend.legend(handles=handles, loc='upper left', frameon=False, labelspacing=1.5, fontsize=title_fontsize, bbox_to_anchor=(-1.7, 1))
    ax_legend.axis('off')

    for i in range(3):
        mask_low  = low_layer  == i
        mask_high = high_layer == i

        gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[i, 0], hspace=0.5, wspace=0.35)

        # low res measured energy                
        ax1 = fig.add_subplot(gs[0])
        low_counts, low_xedges, low_yedges, im1 = ax1.hist2d(
            low_eta[mask_low], low_phi[mask_low], weights=e_trans(low_e_meas[mask_low]), 
            bins=[np.linspace(-3,3,self.low_gran[i]+1), np.linspace(-np.pi,np.pi,self.low_gran[i]+1)],
            cmap=cmap, vmin=vmin, vmax=vmax)
        ax1.set_title("Energy", fontsize=title_fontsize)
        ax1.annotate(f'ECAL{i+1}', xy=(0, 0.5), xytext=(-ax1.yaxis.labelpad - 5, 0),
            xycoords=ax1.yaxis.label, textcoords='offset points',
            ha='right', va='center', fontsize=22, rotation=90)
                
        if i == 0:
            cax = fig.add_axes([0.04, 0.11, 0.007, 0.77])
            cbar = fig.colorbar(im1, cax=cax)
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.ax.tick_params(labelsize=12)
            
            # Set the label at the top and rotate it
            cbar.set_label('$ln \\left( E + 1 \\right)$', labelpad=5, fontsize=18)
            cbar.ax.yaxis.set_label_position('left')
            # cbar.ax.yaxis.label.set_rotation(0)
            cbar.ax.yaxis.label.set_horizontalalignment('center')
            cbar.ax.yaxis.label.set_verticalalignment('bottom')  


        # low res particle flow
        ax2 = fig.add_subplot(gs[1])
        colors = np.ones((*low_counts.shape, 3))
    
        e_mask = low_e_meas > 1
        mask_low_after_e_cut = mask_low[e_mask]

        eta_coord = np.digitize(low_eta[e_mask][mask_low_after_e_cut], low_xedges) - 1
        phi_coord = np.digitize(low_phi[e_mask][mask_low_after_e_cut], low_yedges) - 1
        colors[phi_coord, eta_coord] = attn_rgbs_low[mask_low_after_e_cut]

        ax2.imshow(colors, extent=[low_xedges[0], low_xedges[-1], low_yedges[0], low_yedges[-1]], 
                origin='lower', aspect='auto')
        ax2.set_title("PFlow", fontsize=title_fontsize)


        gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[i, 1], hspace=0.5, wspace=0.35)

        # high res pred energy
        ax3 = fig.add_subplot(gs[0])
        high_counts, high_xedges, high_yedges, _ = ax3.hist2d(
            high_eta[mask_high], high_phi[mask_high], weights=e_trans(high_e_pred[mask_high]), 
            bins=[np.linspace(-3,3,self.high_gran[i]+1), np.linspace(-np.pi,np.pi,self.high_gran[i]+1)],
            cmap=cmap, vmin=vmin, vmax=vmax)
        ax3.set_title(f"Energy", fontsize=title_fontsize)

        # high res particle flow
        ax4 = fig.add_subplot(gs[1])
        colors = np.ones((*high_counts.shape, 3))

        e_mask = high_e_pred > 1
        mask_high_after_e_cut = mask_high[e_mask]

        eta_coord = np.digitize(high_eta[e_mask][mask_high_after_e_cut], high_xedges) - 1
        phi_coord = np.digitize(high_phi[e_mask][mask_high_after_e_cut], high_yedges) - 1
        colors[phi_coord, eta_coord, :] = attn_rgbs_high[mask_high_after_e_cut]

        ax4.imshow(colors, extent=[high_xedges[0], high_xedges[-1], high_yedges[0], high_yedges[-1]], 
                origin='lower', aspect='auto')
        ax4.set_title("PFlow", fontsize=title_fontsize)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlim([xmin, xmax]); ax.set_ylim([ymin, ymax])
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_xlabel(r'$\eta$', fontsize=18)
            ax.set_ylabel(r'$\phi$', fontsize=18)


        if i == 0:
            # ax1.annotate('LR (measured)', xy=(1.18, 1.3), xytext=(1.18, 1.4), xycoords='axes fraction', 
            #     fontsize=22, ha='center', va='bottom',
            #     arrowprops=dict(arrowstyle='-[, widthB=11.5, lengthB=0.5', lw=1.0, color='r'))

            # ax3.annotate('HR (predicted)', xy=(1.18, 1.3), xytext=(1.18, 1.4), xycoords='axes fraction', 
            #     fontsize=22, ha='center', va='bottom',
            #     arrowprops=dict(arrowstyle='-[, widthB=11.5, lengthB=0.5', lw=1.0, color='r'))

            def add_bracket(ax, x0, x1, y, text, text_y_offset, fontsize=12):

                height = 0.01

                # Draw the vertical parts of the bracket
                fig.add_artist(Line2D([x0, x0], [y, y + height], transform=fig.transFigure, color='k', lw=1.0))
                fig.add_artist(Line2D([x1, x1], [y, y + height], transform=fig.transFigure, color='k', lw=1.0))

                # Draw the horizontal part of the bracket
                fig.add_artist(Line2D([x0, x1], [y + height, y + height], transform=fig.transFigure, color='k', lw=1.0))

                # Add the text label
                fig.text((x0 + x1) / 2, y + text_y_offset, text, ha="center", va="bottom", fontsize=fontsize)


            # Add horizontal square bracket for LR
            add_bracket(ax1, 0.113, 0.446, 0.93, "LR (measured)", 0.025, fontsize=22)

            plus = 0.3725

            # Add horizontal square bracket for HR
            add_bracket(ax3, 0.113 + plus, 0.446 + plus, 0.93, "HR (predicted)", 0.025, fontsize=22)




    if dir is None:
        return fig
    
    plt.savefig(os.path.join(dir, f'ED_{idx}.png'))





def plot_pf_event_display_v2(self, idx, dir=None, verbose=False):

    def e_trans(e):
        e = np.clip(e, 0, None)
        return np.log(e+1)

    def get_text(x):
        text = ''
        if len(x) > 0:
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

    if (low_e_meas > 1).sum() != self.inc_wt_lr_pf[0][idx].shape[0]:
        print(f'Low res measured energy shape (', low_e_meas.shape, ') != low res PF energy shape (', self.inc_wt_lr_pf[0][idx].shape, '). Skipping this event.')
        return

    if (high_e_pred > 1).sum() != self.inc_wt_hr_pf[0][idx].shape[0]:
        print((high_e_pred > 1).sum(), self.inc_wt_hr_pf[0][idx].shape[0])
        print('High res truth energy shape (', high_e_truth.shape, ') != high res PF energy shape (', self.inc_wt_hr_pf[0][idx].shape, '). Skipping this event.')
        return

    attn_rgbs_high = self.get_rgb('hr', idx)
    attn_rgbs_low = self.get_rgb('lr', idx)
    attn_rgbs_high_argmax = self.get_rgb('hr', idx, argmax=True)
    attn_rgbs_low_argmax = self.get_rgb('lr', idx, argmax=True)

    vmin = min(high_e_truth.min(), high_e_pred.min(), low_e_meas.min())
    vmax = max(high_e_truth.max(), high_e_pred.max(), low_e_meas.max())
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

    fig = plt.figure(figsize=(21, 13), dpi=300)
    gs_outer = GridSpec(3, 3, hspace=0.43, wspace=0.3, width_ratios=[1, 1, 0.1])
    # fig.suptitle(f'Event {idx}; truth card {len(self.truth_part_dep_e[idx])}', fontsize=16)

    # legend
    gs_legend = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[:, 2], height_ratios=[1, 2])
    ax_legend = fig.add_subplot(gs_legend[0])
    ax_legend.axis('off')  # Hide the axes

    colors = [self.pf_colors[0], self.pf_colors[2], self.pf_colors[1]]
    labels = ['Particle 1', 'Particle 2', 'Particle 3']

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=30, label=label) for color, label in zip(colors, labels)]
    ax_legend.legend(handles=handles, loc='upper left', frameon=False, labelspacing=1.5, fontsize=title_fontsize, bbox_to_anchor=(-1.7, 1))
    ax_legend.axis('off')

    for i in range(3):
        mask_low  = low_layer  == i
        mask_high = high_layer == i

        gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[i, 0], hspace=0.5, wspace=0.35)

        # low res measured energy                
        ax1 = fig.add_subplot(gs[0])
        low_counts, low_xedges, low_yedges, im1 = ax1.hist2d(
            low_eta[mask_low], low_phi[mask_low], weights=e_trans(low_e_meas[mask_low]), 
            bins=[np.linspace(-3,3,self.low_gran[i]+1), np.linspace(-np.pi,np.pi,self.low_gran[i]+1)],
            cmap=cmap, vmin=vmin, vmax=vmax)
        ax1.set_title("Energy", fontsize=title_fontsize)
        ax1.annotate(f'ECAL{i+1}', xy=(0, 0.5), xytext=(-ax1.yaxis.labelpad - 5, 0),
            xycoords=ax1.yaxis.label, textcoords='offset points',
            ha='right', va='center', fontsize=22, rotation=90)
                
        if i == 0:
            cax = fig.add_axes([0.04, 0.11, 0.007, 0.77])
            cbar = fig.colorbar(im1, cax=cax)
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.ax.tick_params(labelsize=12)
            
            # Set the label at the top and rotate it
            cbar.set_label('$ln \\left( E + 1 \\right)$', labelpad=5, fontsize=18)
            cbar.ax.yaxis.set_label_position('left')
            # cbar.ax.yaxis.label.set_rotation(0)
            cbar.ax.yaxis.label.set_horizontalalignment('center')
            cbar.ax.yaxis.label.set_verticalalignment('bottom')  


        # low res particle flow
        ax2 = fig.add_subplot(gs[1])
        colors = np.ones((*low_counts.shape, 3))
    
        e_mask = low_e_meas > 1
        mask_low_after_e_cut = mask_low[e_mask]

        eta_coord = np.digitize(low_eta[e_mask][mask_low_after_e_cut], low_xedges) - 1
        phi_coord = np.digitize(low_phi[e_mask][mask_low_after_e_cut], low_yedges) - 1
        colors[phi_coord, eta_coord] = attn_rgbs_low[mask_low_after_e_cut]

        ax2.imshow(colors, extent=[low_xedges[0], low_xedges[-1], low_yedges[0], low_yedges[-1]], 
                origin='lower', aspect='auto')
        ax2.set_title("PFlow", fontsize=title_fontsize)


        gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[i, 1], hspace=0.5, wspace=0.35)

        # high res pred energy
        ax3 = fig.add_subplot(gs[0])
        high_counts, high_xedges, high_yedges, _ = ax3.hist2d(
            high_eta[mask_high], high_phi[mask_high], weights=e_trans(high_e_pred[mask_high]), 
            bins=[np.linspace(-3,3,self.high_gran[i]+1), np.linspace(-np.pi,np.pi,self.high_gran[i]+1)],
            cmap=cmap, vmin=vmin, vmax=vmax)
        ax3.set_title(f"Energy", fontsize=title_fontsize)

        # high res particle flow
        ax4 = fig.add_subplot(gs[1])
        colors = np.ones((*high_counts.shape, 3))

        e_mask = high_e_pred > 1
        mask_high_after_e_cut = mask_high[e_mask]

        eta_coord = np.digitize(high_eta[e_mask][mask_high_after_e_cut], high_xedges) - 1
        phi_coord = np.digitize(high_phi[e_mask][mask_high_after_e_cut], high_yedges) - 1
        colors[phi_coord, eta_coord, :] = attn_rgbs_high[mask_high_after_e_cut]

        ax4.imshow(colors, extent=[high_xedges[0], high_xedges[-1], high_yedges[0], high_yedges[-1]], 
                origin='lower', aspect='auto')
        ax4.set_title("PFlow", fontsize=title_fontsize)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlim([xmin, xmax]); ax.set_ylim([ymin, ymax])
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_xlabel(r'$\eta$', fontsize=18)
            ax.set_ylabel(r'$\phi$', fontsize=18)


        if i == 0:
            # ax1.annotate('LR (measured)', xy=(1.18, 1.3), xytext=(1.18, 1.4), xycoords='axes fraction', 
            #     fontsize=22, ha='center', va='bottom',
            #     arrowprops=dict(arrowstyle='-[, widthB=11.5, lengthB=0.5', lw=1.0, color='r'))

            # ax3.annotate('HR (predicted)', xy=(1.18, 1.3), xytext=(1.18, 1.4), xycoords='axes fraction', 
            #     fontsize=22, ha='center', va='bottom',
            #     arrowprops=dict(arrowstyle='-[, widthB=11.5, lengthB=0.5', lw=1.0, color='r'))

            def add_bracket(ax, x0, x1, y, text, text_y_offset, fontsize=12):

                height = 0.01

                # Draw the vertical parts of the bracket
                fig.add_artist(Line2D([x0, x0], [y, y + height], transform=fig.transFigure, color='k', lw=1.0))
                fig.add_artist(Line2D([x1, x1], [y, y + height], transform=fig.transFigure, color='k', lw=1.0))

                # Draw the horizontal part of the bracket
                fig.add_artist(Line2D([x0, x1], [y + height, y + height], transform=fig.transFigure, color='k', lw=1.0))

                # Add the text label
                fig.text((x0 + x1) / 2, y + text_y_offset, text, ha="center", va="bottom", fontsize=fontsize)


            # Add horizontal square bracket for LR
            add_bracket(ax1, 0.113, 0.446, 0.93, "LR (measured)", 0.025, fontsize=22)

            plus = 0.3725

            # Add horizontal square bracket for HR
            add_bracket(ax3, 0.113 + plus, 0.446 + plus, 0.93, "HR (predicted)", 0.025, fontsize=22)




    if dir is None:
        return fig
    
    plt.savefig(os.path.join(dir, f'ED_{idx}.png'))




def plot_pf_event_display_old(self, idx, dir=None, verbose=False):

    def e_trans(e):
        e = np.clip(e, 0, None)
        return np.log(e+1)

    def get_text(x):
        text = ''
        if len(x) > 0:
            text = f'sum = {x.sum():.0f} MeV \npeak = {x.max():.0f} MeV'
        return text

    cmap = mpl.cm.get_cmap("plasma_r").copy()
    cmap.set_under(color='white')

    low_eta    = self.low_eta[idx]
    low_phi    = self.low_phi[idx]
    low_layer  = self.low_layer[idx]
    low_e_meas = self.low_e_measured[idx]

    high_eta    = self.high_eta[idx]
    high_phi    = self.high_phi[idx]
    high_layer  = self.high_layer[idx]
    high_e_pred  = self.high_e_pred[idx]
    high_e_truth = self.high_e_truth[idx]

    if (low_e_meas > 1).sum() != self.inc_wt_lr_pf[0][idx].shape[0]:
        print(f'Low res measured energy shape (', low_e_meas.shape, ') != low res PF energy shape (', self.inc_wt_lr_pf[0][idx].shape, '). Skipping this event.')
        return

    if (high_e_pred > 1).sum() != self.inc_wt_hr_pf[0][idx].shape[0]:
        print((high_e_pred > 1).sum(), self.inc_wt_hr_pf[0][idx].shape[0])
        print('High res truth energy shape (', high_e_truth.shape, ') != high res PF energy shape (', self.inc_wt_hr_pf[0][idx].shape, '). Skipping this event.')
        return

    attn_rgbs_high = self.get_rgb('hr', idx)
    attn_rgbs_low = self.get_rgb('lr', idx)
    attn_rgbs_high_argmax = self.get_rgb('hr', idx, argmax=True)
    attn_rgbs_low_argmax = self.get_rgb('lr', idx, argmax=True)

    vmin = min(high_e_truth.min(), high_e_pred.min(), low_e_meas.min())
    vmax = max(high_e_truth.max(), high_e_pred.max(), low_e_meas.max())
    vmin, vmax = 1, e_trans(vmax)

    xmin, xmax = \
        min(low_eta.min(), high_eta.min()), max(low_eta.max(), high_eta.max())
    ymin, ymax = \
        min(low_phi.min(), high_phi.min()), max(low_phi.max(), high_phi.max())
        
    xrange, yrange = xmax - xmin, ymax - ymin

    xmin, xmax = xmin - xrange/5, xmax + xrange/5
    ymin, ymax = ymin - yrange/5, ymax + yrange/5

    xgrid_, ygrid_ = np.linspace(-3,3,64), np.linspace(-np.pi,np.pi,64)
    
    xmin = xgrid_[np.abs(xgrid_ - (xmin - (xgrid_[1]-xgrid_[0]))).argmin()]
    xmax = xgrid_[np.abs(xgrid_ - (xmax + (xgrid_[1]-xgrid_[0]))).argmin()]
    ymin = ygrid_[np.abs(ygrid_ - (ymin - (ygrid_[1]-ygrid_[0]))).argmin()]
    ymax = ygrid_[np.abs(ygrid_ - (ymax + (ygrid_[1]-ygrid_[0]))).argmin()]

    ncol = 9
    fig = plt.figure(figsize=(3*ncol, 3*3), dpi=100) # dpi=200
    gs = GridSpec(3, ncol, hspace=0.5, wspace=0.3, width_ratios=[1,1,1, 0.2, 1,1, 0.1, 1,1])
    fig.suptitle(f'Event {idx}', fontsize=16)

    for i in range(3):
        mask_low  = low_layer  == i
        mask_high = high_layer == i

        # low res measured energy                
        ax1 = fig.add_subplot(gs[i*ncol])
        low_counts, low_xedges, low_yedges, im1 = ax1.hist2d(
            low_eta[mask_low], low_phi[mask_low], weights=e_trans(low_e_meas[mask_low]), 
            bins=[np.linspace(-3,3,self.low_gran[i]+1), np.linspace(-np.pi,np.pi,self.low_gran[i]+1)],
            cmap=cmap, vmin=vmin, vmax=vmax)
        ax1.text(0.05, 0.8, get_text(low_e_meas[mask_low]), transform=ax1.transAxes)
        ax1.set_title("LR (meas E)")
        if i == 0:
            cax = fig.add_axes([0.09, 0.2, 0.005, 0.6])
            cbar = fig.colorbar(im1, cax=cax)
            cbar.ax.yaxis.set_ticks_position('left')

        # high res truth energy
        ax2 = fig.add_subplot(gs[i*ncol + 1])
        ax2.hist2d(
            high_eta[mask_high], high_phi[mask_high], weights=e_trans(high_e_truth[mask_high]),
            bins=[np.linspace(-3,3,self.high_gran[i]+1), np.linspace(-np.pi,np.pi,self.high_gran[i]+1)],
            cmap=cmap, vmin=vmin, vmax=vmax)
        ax2.text(0.05, 0.8, get_text(high_e_truth[mask_high]), transform=ax2.transAxes)
        ax2.set_title(f"HR (truth E)")

        # high res pred energy
        ax3 = fig.add_subplot(gs[i*ncol + 2])
        high_counts, high_xedges, high_yedges, im3 = ax3.hist2d(
            high_eta[mask_high], high_phi[mask_high], weights=e_trans(high_e_pred[mask_high]), 
            bins=[np.linspace(-3,3,self.high_gran[i]+1), np.linspace(-np.pi,np.pi,self.high_gran[i]+1)],
            cmap=cmap, vmin=vmin, vmax=vmax)
        ax3.text(0.05, 0.8, get_text(high_e_pred[mask_high]), transform=ax3.transAxes)
        ax3.set_title(f"HR (pred E)")

        
        # low res particle flow
        ax4 = fig.add_subplot(gs[i*ncol + 4])
        colors = np.ones((*low_counts.shape, 3))
    
        e_mask = low_e_meas > 1
        mask_low_after_e_cut = mask_low[e_mask]

        eta_coord = np.digitize(low_eta[e_mask][mask_low_after_e_cut], low_xedges) - 1
        phi_coord = np.digitize(low_phi[e_mask][mask_low_after_e_cut], low_yedges) - 1
        colors[phi_coord, eta_coord] = attn_rgbs_low[mask_low_after_e_cut]

        ax4.imshow(colors, extent=[low_xedges[0], low_xedges[-1], low_yedges[0], low_yedges[-1]], 
                origin='lower', aspect='auto')
        ax4.set_title("PF (LR meas)")

        # high res particle flow
        ax5 = fig.add_subplot(gs[i*ncol + 5])
        colors = np.ones((*high_counts.shape, 3))

        e_mask = high_e_pred > 1
        mask_high_after_e_cut = mask_high[e_mask]

        eta_coord = np.digitize(high_eta[e_mask][mask_high_after_e_cut], high_xedges) - 1
        phi_coord = np.digitize(high_phi[e_mask][mask_high_after_e_cut], high_yedges) - 1
        colors[phi_coord, eta_coord, :] = attn_rgbs_high[mask_high_after_e_cut]

        ax5.imshow(colors, extent=[high_xedges[0], high_xedges[-1], high_yedges[0], high_yedges[-1]], 
                origin='lower', aspect='auto')
        ax5.set_title("PF (HR pred)")

        # low res particle flow (argmax)
        ax6 = fig.add_subplot(gs[i*ncol + 7])
        colors = np.ones((*low_counts.shape, 3))

        e_mask = low_e_meas > 1
        mask_low_after_e_cut = mask_low[e_mask]

        eta_coord = np.digitize(low_eta[e_mask][mask_low_after_e_cut], low_xedges) - 1
        phi_coord = np.digitize(low_phi[e_mask][mask_low_after_e_cut], low_yedges) - 1
        colors[phi_coord, eta_coord] = attn_rgbs_low_argmax[mask_low_after_e_cut]

        ax6.imshow(colors, extent=[low_xedges[0], low_xedges[-1], low_yedges[0], low_yedges[-1]],
                origin='lower', aspect='auto')
        ax6.set_title("PF (LR meas) argmax")

        # high res particle flow (argmax)
        ax7 = fig.add_subplot(gs[i*ncol + 8])
        colors = np.ones((*high_counts.shape, 3))

        e_mask = high_e_pred > 1
        mask_high_after_e_cut = mask_high[e_mask]

        eta_coord = np.digitize(high_eta[e_mask][mask_high_after_e_cut], high_xedges) - 1
        phi_coord = np.digitize(high_phi[e_mask][mask_high_after_e_cut], high_yedges) - 1
        colors[phi_coord, eta_coord, :] = attn_rgbs_high_argmax[mask_high_after_e_cut]

        ax7.imshow(colors, extent=[high_xedges[0], high_xedges[-1], high_yedges[0], high_yedges[-1]],
                origin='lower', aspect='auto')
        ax7.set_title("PF (HR pred) argmax")

        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
            ax.set_xlim([xmin, xmax]); ax.set_ylim([ymin, ymax])

    if verbose:
        print()
        print('event idx:', idx)
        print('truth_part_dep_e [GeV]:', self.truth_part_dep_e[idx] * 1e-3)
        print('low_part_e [GeV]:', self.low_part_e[idx] * 1e-3)
        print('high_part_e [GeV]:', self.high_part_e[idx] * 1e-3)
        print()
        print('truth_part_eta:', self.truth_part_eta[idx])
        print('low_part_eta:', self.low_part_eta[idx])
        print('high_part_eta:', self.high_part_eta[idx])
        print()
        print('truth_part_phi:', self.truth_part_phi[idx])
        print('low_part_phi:', self.low_part_phi[idx])
        print('high_part_phi:', self.high_part_phi[idx])

    if dir is None:
        return fig
    
    plt.savefig(os.path.join(dir, f'ED_{idx}.png'))

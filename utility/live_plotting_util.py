import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import utility.transformation as trans
from performance import PerformanceCOCOA

def graph2img_scd(graph_dict, fig=None):
    
    def get_text_sum_peak(x):
        return f'sum = {x.sum():.0f} MeV \npeak = {x.max():.0f} MeV'

    def get_text_min_max(x):
        return f'min = {x.min():.2f} \nmax = {x.max():.2f}'

    # used on the evaluated graph
    low_gran  = [128, 128,  64, 32, 32, 16]
    high_gran = [256, 256, 128, 64, 64, 32]

    eta    = graph_dict['eta_raw']
    phi    = graph_dict['phi']
    layer  = graph_dict['layer']

    target_nn = graph_dict['target']
    pred_nn   = graph_dict['pred']

    # we want them in MeV (better looking plots)
    e_truth_raw = np.clip(graph_dict['e_truth_raw'], 0, None) * 1e3
    e_pred_raw  = np.clip(graph_dict['e_pred_raw'], 0, None) * 1e3
    log_e_truth_raw = np.log(e_truth_raw + 1)
    log_e_pred_raw  = np.log(e_pred_raw + 1)

    vmin = 1e-8; vmax = max(log_e_truth_raw.max(), log_e_pred_raw.max())
    vmin_raw_nn = min(target_nn.min(), pred_nn.min())
    vmax_raw_nn = max(target_nn.max(), pred_nn.max())

    xmin, xmax = eta.min(), eta.max()
    ymin, ymax = phi.min(), phi.max()
        
    xrange, yrange = xmax - xmin, ymax - ymin
        
    xmin, xmax = xmin - xrange/8, xmax + xrange/8
    ymin, ymax = ymin - yrange/8, ymax + yrange/8
    
    xgrid_, ygrid_ = np.linspace(-3,3,64), np.linspace(-np.pi,np.pi,64)
    
    xmin = xgrid_[np.abs(xgrid_ - (xmin - (xgrid_[1]-xgrid_[0]))).argmin()]
    xmax = xgrid_[np.abs(xgrid_ - (xmax + (xgrid_[1]-xgrid_[0]))).argmin()]
    ymin = ygrid_[np.abs(ygrid_ - (ymin - (ygrid_[1]-ygrid_[0]))).argmin()]
    ymax = ygrid_[np.abs(ygrid_ - (ymax + (ygrid_[1]-ygrid_[0]))).argmin()]

    cmap = mpl.cm.get_cmap("plasma_r").copy()
    cmap.set_under(color='white')

    if fig == None:
        fig = plt.figure(figsize=(15,30), dpi=300)
        
    for i in range(3):

        mask = layer == i
        
        # if we don't have any entry for this layer, skip
        if mask.sum() == 0:
            continue

        bins = [np.linspace(-3,3,high_gran[i]+1), np.linspace(-np.pi,np.pi,high_gran[i]+1)]

        ax1 = fig.add_subplot(3,5,i*5+1)
        heatmap1 = ax1.hist2d(
            eta[mask], phi[mask], weights=log_e_truth_raw[mask], 
            bins=bins, vmin=vmin, vmax=vmax, cmap=cmap)
        fig.colorbar(heatmap1[3], ax=ax1)
        ax1.text(0.05, 0.8, get_text_sum_peak(e_truth_raw[mask]), transform=ax1.transAxes)
        ax1.set_title(f'$log(E_{{True}} + 1)$ layer {i}')
                
        ax2 = fig.add_subplot(3,5,i*5+2)
        heatmap2 =ax2.hist2d(
            eta[mask], phi[mask], weights=log_e_pred_raw[mask], 
            bins=bins, vmin=vmin, vmax=vmax, cmap=cmap)
        fig.colorbar(heatmap2[3], ax=ax2)
        ax2.text(0.05, 0.8, get_text_sum_peak(e_pred_raw[mask]), transform=ax2.transAxes)
        ax2.set_title(f'$log(E_{{Pred}} + 1)$ layer {i}')


        bin_x_indices = np.digitize(eta[mask], bins[0]) - 1
        bin_y_indices = np.digitize(phi[mask], bins[1]) - 1

        bin_centers_eta = (bins[0][:-1] + bins[0][1:]) / 2
        bin_centers_phi = (bins[1][:-1] + bins[1][1:]) / 2
        bin_centers_eta, bin_centers_phi = np.meshgrid(bin_centers_eta, bin_centers_phi)
        bin_centers_eta, bin_centers_phi = bin_centers_eta.T.flatten(), bin_centers_phi.T.flatten()

        weight_matrix = np.zeros((len(bins[0]) - 1, len(bins[1]) - 1)) - 1000
        weight_matrix[bin_x_indices, bin_y_indices] = target_nn[mask]

        ax3 = fig.add_subplot(3,5,i*5+3)
        heatmap3 = ax3.hist2d(
            bin_centers_eta, bin_centers_phi, weights=weight_matrix.flatten(),
            bins=bins, vmin=vmin_raw_nn, vmax=vmax_raw_nn, cmap=cmap)
        fig.colorbar(heatmap3[3], ax=ax3)
        ax3.text(0.05, 0.8, get_text_min_max(target_nn[mask]), transform=ax3.transAxes)
        ax3.set_title(f'Target (NN) layer {i}')

        weight_matrix = np.zeros((len(bins[0]) - 1, len(bins[1]) - 1)) - 1000
        weight_matrix[bin_x_indices, bin_y_indices] = pred_nn[mask]

        ax4 = fig.add_subplot(3,5,i*5+4)
        heatmap4 = ax4.hist2d(
            bin_centers_eta, bin_centers_phi, weights=weight_matrix.flatten(),
            bins=bins, vmin=vmin_raw_nn, vmax=vmax_raw_nn, cmap=cmap)
        fig.colorbar(heatmap4[3], ax=ax4)
        ax4.text(0.05, 0.8, get_text_min_max(pred_nn[mask]), transform=ax4.transAxes)
        ax4.set_title(f'Pred (NN) layer {i}')
        
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlim([xmin, xmax]); ax.set_ylim([ymin, ymax])
            ax.set_xlabel('$\eta$'); ax.set_ylabel('$\phi$')
            ax.set_aspect('auto',  adjustable='box')


    ax1 = fig.add_subplot(3,5,5)
    e_raw_diff = e_pred_raw - e_truth_raw
    ax1.hlines(0, e_truth_raw.min(), e_truth_raw.max(), color='black', lw=0.5)
    ax1.scatter(e_truth_raw, e_raw_diff, s=5, color='purple', alpha=0.5)
    ax1.set_ylabel('$E_{pred} - E_{true}$ [MeV]')
    ax1.set_xlabel('$E_{true}$ [MeV]')

    ax2 = fig.add_subplot(3,5,10)
    nn_diff = pred_nn - target_nn
    ax2.hlines(0, e_truth_raw.min(), e_truth_raw.max(), color='black', lw=0.5)
    ax2.scatter(e_truth_raw, nn_diff, s=5, color='purple', alpha=0.5)
    ax2.set_ylabel('Pred (NN) - Target (NN)')
    ax2.set_xlabel('$E_{true}$ [MeV]')

    ax3 = fig.add_subplot(3,5,15)
    ax3.hlines(0, target_nn.min(), target_nn.max(), color='black', lw=0.5)
    ax3.scatter(target_nn, nn_diff, s=5, color='purple', alpha=0.5)
    ax3.set_ylabel('Pred (NN) - Target (NN)')
    ax3.set_xlabel('Target (NN)')

    return fig


class PerformanceCOCOALive(PerformanceCOCOA):

    def __init__(self, res_factor, trans_dict, target_trans_obj):
        self.res_factor = res_factor

        self.high_gran = [256, 256, 128, 64, 64, 32]
        if res_factor == 2:
            self.low_gran  = [128, 128,  64, 32, 32, 16]
        elif res_factor == 4:
            self.low_gran  = [ 64,  64,  32, 16, 16,  8]
        else:
            raise ValueError('res_factor must be 2 or 4')
        
        self.trans_dict = trans_dict
        self.target_trans_obj = target_trans_obj

        self.reset()


    def reset(self):

        self.n_events = 0

        self.low_phi = []
        self.low_eta = []
        self.low_layer = []
        self.low_e_measured = []

        self.high_phi = []
        self.high_eta = []
        self.high_layer = []

        self.high_e_proxy = []
        self.high_e_pred = []
        self.high_e_truth = []


    def update(self, batch, pred):
        self.n_events += batch['q_mask'].shape[0]

        for bs_i in range(batch['q_mask'].shape[0]):
            q_mask_bs_i = batch['q_mask'][bs_i]
            low_q_mask_bs_i = batch['low_q_mask'][bs_i]

            self.low_phi.append(batch['low_phi'][bs_i][low_q_mask_bs_i].squeeze(-1).detach().cpu().numpy())
            self.low_eta.append(batch['low_eta_raw'][bs_i][low_q_mask_bs_i].squeeze(-1).detach().cpu().numpy())
            self.low_layer.append(batch['low_layer'][bs_i][low_q_mask_bs_i].squeeze(-1).detach().cpu().numpy())
            self.low_e_measured.append(batch['low_e_meas_raw'][bs_i][low_q_mask_bs_i].squeeze(-1).detach().cpu().numpy() * 1e3)

            self.high_phi.append(batch['phi'][bs_i][q_mask_bs_i].squeeze(-1).detach().cpu().numpy())
            self.high_eta.append(batch['eta_raw'][bs_i][q_mask_bs_i].squeeze(-1).detach().cpu().numpy())
            self.high_layer.append(batch['layer'][bs_i][q_mask_bs_i].squeeze(-1).detach().cpu().numpy())

            self.high_e_proxy.append(batch['e_proxy_raw'][bs_i][q_mask_bs_i].squeeze(-1).detach().cpu().numpy() * 1e3)
            self.high_e_truth.append(batch['e_truth_raw'][bs_i][q_mask_bs_i].squeeze(-1).detach().cpu().numpy() * 1e3)
            
            pred_bs_i = pred[bs_i][q_mask_bs_i].squeeze(-1)
            e_pred_raw = self.target_trans_obj.inverse(pred_bs_i, batch['e_proxy_raw'][bs_i][q_mask_bs_i].squeeze(-1))
            self.high_e_pred.append(e_pred_raw.detach().cpu().numpy() * 1e3)

import uproot
import numpy as np
import torch
import dgl

from torch.utils.data import Dataset

from utility.transformation import VarTransformation
from utility.target_transformation import TargetTransformation



class SupResDataset(Dataset):
    def __init__(self, filename, config_mv=None, make_low_graph=False, make_particle_graph=False,
                 entry_start=0, reduce_ds=-1, one_event_train=False, one_event_idx=0):

        # reading the dataset
        self.config_mv = config_mv
        self.var_transform = self.config_mv['var_transform']
        self.res_factor = self.config_mv['res_factor']
        self.make_low_graph = make_low_graph
        self.make_particle_graph = make_particle_graph
        self.one_event_train = one_event_train
        self.one_event_idx = one_event_idx

        self.f = uproot.open(filename)

        self.tree_low  = self.f['Low_Tree']
        self.tree_high = self.f['High_Tree']

        self.nevents = self.tree_low.num_entries
        if reduce_ds != -1:
            if reduce_ds < 1:
                self.nevents = int(self.nevents * reduce_ds)
            else:
                self.nevents = min(reduce_ds, self.nevents)
        entry_stop = entry_start + self.nevents


        self.data_dict = {}
        var_list = [
            'cell_eta', 'cell_phi', 'cell_layer', 'cell_e',
            'cell_to_cell_edge_start', 'cell_to_cell_edge_end',
            'cell_x', 'cell_y', 'cell_z'
        ]

        #loading the data
        for var in var_list:
            self.data_dict[f'{var}_low']  = self.tree_low[var].array(
                library='np', entry_start=entry_start, entry_stop=entry_stop)
            self.data_dict[f'{var}_high'] = self.tree_high[var].array(
                library='np', entry_start=entry_start, entry_stop=entry_stop)
            
        if self.make_particle_graph:
            part_var_list = [
                'particle_pt', 'particle_eta', 'particle_phi', 'particle_e', 'particle_pdgid', 'particle_dep_energy']
            for var in part_var_list:
                self.data_dict[var] = self.tree_low[var].array(
                    library='np', entry_start=entry_start, entry_stop=entry_stop)

            # need these to compute particle dep energy while ignoring hcal cells
            for var in ['particle_to_node_idx', 'particle_to_node_weight']:
                self.data_dict[var] = self.tree_high[var].array(
                    library='np', entry_start=entry_start, entry_stop=entry_stop)
            

        # MeV to GeV
        self.data_dict['cell_e_low']  = self.data_dict['cell_e_low'] * 1.0e-3
        self.data_dict['cell_e_high'] = self.data_dict['cell_e_high'] * 1.0e-3


        # adding cos and sin of phi
        self.data_dict['cell_cosphi_low']  = np.array([np.cos(x) for x in self.data_dict['cell_phi_low']], dtype=object)
        self.data_dict['cell_sinphi_low']  = np.array([np.sin(x) for x in self.data_dict['cell_phi_low']], dtype=object)
        self.data_dict['cell_cosphi_high'] = np.array([np.cos(x) for x in self.data_dict['cell_phi_high']], dtype=object)
        self.data_dict['cell_sinphi_high'] = np.array([np.sin(x) for x in self.data_dict['cell_phi_high']], dtype=object)


        # cardinality
        self.cell_count_low =  [len(x) for x in self.data_dict['cell_eta_low']]
        self.cell_count_high = [len(x) for x in self.data_dict['cell_eta_high']]


        # low to high connections. use it to reorder the high cells
        self.data_dict['high_to_low'] = self.tree_low['high_cell_to_low_cell_edge'].array(
            library='np', entry_start=entry_start, entry_stop=entry_stop)

        print(f'done loading data ({self.nevents} events)')


        # create the transform dict
        self.transform_dicts = {}
        for k, v in self.var_transform.items():
            self.transform_dicts[k] = VarTransformation(v)
        self.target_trans_obj = TargetTransformation(self.config_mv['target_transform'])


    def __getitem__(self, idx):

        # speed hack (train on one events)
        if self.one_event_train:
            idx = self.one_event_idx

        # get vars
        cell_low_eta_raw    = torch.from_numpy(self.data_dict['cell_eta_low'][idx])
        cell_low_phi        = torch.from_numpy(self.data_dict['cell_phi_low'][idx])
        cell_low_cosphi     = torch.from_numpy(self.data_dict['cell_cosphi_low'][idx])
        cell_low_sinphi     = torch.from_numpy(self.data_dict['cell_sinphi_low'][idx])
        cell_low_layer      = torch.from_numpy(self.data_dict['cell_layer_low'][idx])

        cell_low_e_meas_raw = torch.from_numpy(self.data_dict['cell_e_low'][idx])

        reorder_indices  = self.data_dict['high_to_low'][idx]

        cell_high_eta_raw    = torch.from_numpy(self.data_dict['cell_eta_high'][idx][reorder_indices])
        cell_high_phi        = torch.from_numpy(self.data_dict['cell_phi_high'][idx][reorder_indices])
        cell_high_cosphi     = torch.from_numpy(self.data_dict['cell_cosphi_high'][idx][reorder_indices])
        cell_high_sinphi     = torch.from_numpy(self.data_dict['cell_sinphi_high'][idx][reorder_indices])
        cell_high_layer      = torch.from_numpy(self.data_dict['cell_layer_high'][idx][reorder_indices])
        cell_high_e_truth_raw = torch.from_numpy(self.data_dict['cell_e_high'][idx][reorder_indices])

        cell_high_x_raw = torch.from_numpy(self.data_dict['cell_x_high'][idx][reorder_indices])
        cell_high_y_raw = torch.from_numpy(self.data_dict['cell_y_high'][idx][reorder_indices])
        cell_high_z_raw = torch.from_numpy(self.data_dict['cell_z_high'][idx][reorder_indices])


        # make graph
        num_low, num_high = len(cell_low_eta_raw), len(cell_high_eta_raw)
        num_particles = len(self.data_dict['particle_pt'][idx]) if self.make_particle_graph else 0
        num_nodes_dict = {
            'high': num_high,
            'low' : num_low,
            'particle': num_particles,
            'global_node': 1
        }

        if self.config_mv['graph_building'] ==  'predefined':
            high_to_high_start = self.data_dict['cell_to_cell_edge_start_high'][idx]
            high_to_high_end   = self.data_dict['cell_to_cell_edge_end_high'][idx]

        elif self.config_mv['graph_building'] == 'all2all':
            high_to_high_start, high_to_high_end = torch.meshgrid(
                torch.arange(num_high), torch.arange(num_high), indexing='xy')
            high_to_high_start, high_to_high_end = high_to_high_start.flatten(), high_to_high_end.flatten()

            # # if we don't want self connections
            # mask = high_to_high_start != high_to_high_end
            # high_to_high_start, high_to_high_end = high_to_high_start[mask], high_to_high_end[mask]

        else:
            raise ValueError('Invalid graph_building method')

        data_dict = {
			('high','high_to_high','high') : (high_to_high_start, high_to_high_end)
        }
            
        g = dgl.heterograph(data_dict, num_nodes_dict)

        # low graph
        if self.make_low_graph:
            g.nodes['low'].data['eta_raw']    = cell_low_eta_raw
            g.nodes['low'].data['phi']        = cell_low_phi
            g.nodes['low'].data['cosphi']     = cell_low_cosphi
            g.nodes['low'].data['sinphi']     = cell_low_sinphi
            g.nodes['low'].data['layer']      = cell_low_layer
            g.nodes['low'].data['e_meas_raw'] = cell_low_e_meas_raw

        # high graph
        g.nodes['high'].data['eta_raw']     = cell_high_eta_raw
        g.nodes['high'].data['phi']         = cell_high_phi
        g.nodes['high'].data['cosphi']      = cell_high_cosphi
        g.nodes['high'].data['sinphi']      = cell_high_sinphi
        g.nodes['high'].data['layer']       = cell_high_layer
        g.nodes['high'].data['e_truth_raw'] = cell_high_e_truth_raw

        g.nodes['high'].data['x_raw'] = cell_high_x_raw
        g.nodes['high'].data['y_raw'] = cell_high_y_raw
        g.nodes['high'].data['z_raw'] = cell_high_z_raw



        # variable transform
        g.nodes['high'].data['x'] = self.transform_dicts['x'].forward(cell_high_x_raw)
        g.nodes['high'].data['y'] = self.transform_dicts['y'].forward(cell_high_y_raw)
        g.nodes['high'].data['z'] = self.transform_dicts['z'].forward(cell_high_z_raw)

        g.nodes['high'].data['eta'] = self.transform_dicts['eta'].forward(cell_high_eta_raw)
        if self.make_low_graph:
            g.nodes['low'].data['eta'] = self.transform_dicts['eta'].forward(cell_low_eta_raw)


        # create the eventwise transformation object
        e_trans_cfg_ev = self.config_mv['var_transform']['e']
        tmp_trans_obj = VarTransformation(e_trans_cfg_ev)
        tmp_trans_cond = tmp_trans_obj.trans(cell_low_e_meas_raw)

        if e_trans_cfg_ev['scale_mode'] == 'min_max':
            e_trans_cfg_ev['min'] = tmp_trans_cond.min()
            e_trans_cfg_ev['max'] = tmp_trans_cond.max()

        elif e_trans_cfg_ev['scale_mode'] == 'standard':
            e_trans_cfg_ev['mean'] = tmp_trans_cond.mean().item()
            e_trans_cfg_ev['std'] = tmp_trans_cond.std().item()

        cond_trans_obj = VarTransformation(e_trans_cfg_ev)


        # transform the energy
        cell_low_e_meas   = cond_trans_obj.forward(cell_low_e_meas_raw)
        cell_high_e_truth = cond_trans_obj.forward(cell_high_e_truth_raw)
        g.nodes['high'].data['e_truth'] = cell_high_e_truth
        if self.make_low_graph:
            g.nodes['low'].data['e_meas'] = cell_low_e_meas

        # proxy energy (interpolated)
        cell_low_e_interp_raw = cell_low_e_meas_raw.repeat_interleave((self.res_factor)**2)
        g.nodes['high'].data['e_proxy_raw'] = cell_low_e_interp_raw
        e_proxy = cond_trans_obj.forward(cell_low_e_interp_raw)
        g.nodes['high'].data['e_proxy'] = e_proxy

        # # target computation (e_truth - e_proxy)
        # target = cell_high_e_truth # - g.nodes['high'].data['e_proxy']
        # g.nodes['high'].data['target'] = target

        target = self.target_trans_obj.forward(hr_truth_raw=cell_high_e_truth_raw, proxy_raw=cell_low_e_interp_raw)
        g.nodes['high'].data['target'] = target


        if self.make_particle_graph:
            g.nodes['particle'].data['pt'] = torch.from_numpy(self.data_dict['particle_pt'][idx])
            g.nodes['particle'].data['eta'] = torch.from_numpy(self.data_dict['particle_eta'][idx])
            g.nodes['particle'].data['phi'] = torch.from_numpy(self.data_dict['particle_phi'][idx])
            g.nodes['particle'].data['e'] = torch.from_numpy(self.data_dict['particle_e'][idx])
            g.nodes['particle'].data['pdgid'] = torch.from_numpy(self.data_dict['particle_pdgid'][idx]).int()
            

            # get the high resolution incidence matrix (w/o reordering the cells)
            part_to_node_idx = self.data_dict['particle_to_node_idx'][idx]
            part_to_node_weight = self.data_dict['particle_to_node_weight'][idx]
            part_dep_e = torch.from_numpy(self.data_dict['particle_dep_energy'][idx])

            # sum over cell for a particle is one
            weight_matrix = torch.zeros(num_high, num_particles)
            for pi, (cell_idxs, cell_wts) in enumerate(zip(part_to_node_idx, part_to_node_weight)):
                pi_inv_attnenuation = 2 if abs(g.nodes['particle'].data['pdgid'][pi]) == 11 else 1
                for ci, cw in zip(cell_idxs, cell_wts):
                    if ci >= num_high:
                        continue
                    weight_matrix[int(ci), pi] = cw * pi_inv_attnenuation
            
            # reorder the cells
            weight_matrix = weight_matrix[reorder_indices]

            # convert to energy matrix
            energy_matrix = weight_matrix * part_dep_e.view(1, num_particles)

            # high energy matrix
            for i in range(num_particles):
                g.nodes['high'].data[f'e_part_{i}'] = energy_matrix[:, i]

            # low energy matrix
            energy_matrix_low = \
                energy_matrix.view(num_low, self.res_factor**2, num_particles).sum(dim=1)
            for i in range(num_particles):
                g.nodes['low'].data[f'e_part_{i}'] = energy_matrix_low[:, i]

            # particle dep_e (ignoring hcal cells)
            g.nodes['particle'].data['dep_e'] = energy_matrix[cell_high_layer < 3].sum(dim=0)


        # remove the cells not in the ecal
        ecal_high_mask = cell_high_layer < 3
        g.remove_nodes(torch.where(ecal_high_mask == False)[0], ntype='high')

        ecal_low_mask = cell_low_layer < 3
        g.remove_nodes(torch.where(ecal_low_mask == False)[0], ntype='low')


        return g, cond_trans_obj, idx


    def __len__(self):
        return self.nevents 

    

def collate_graphs(samples):

    batch_size = len(samples)
    batch_num_nodes = [x[0].number_of_nodes('high') for x in samples]
    max_num_nodes = max(batch_num_nodes)

    eta = torch.zeros(batch_size, max_num_nodes)
    phi = torch.zeros(batch_size, max_num_nodes)
    cosphi = torch.zeros(batch_size, max_num_nodes)
    sinphi = torch.zeros(batch_size, max_num_nodes)
    layer = torch.zeros(batch_size, max_num_nodes, dtype=torch.int)

    e_truth = torch.zeros(batch_size, max_num_nodes)
    e_proxy = torch.zeros(batch_size, max_num_nodes)

    eta_raw     = torch.zeros(batch_size, max_num_nodes)
    e_truth_raw = torch.zeros(batch_size, max_num_nodes)
    e_proxy_raw = torch.zeros(batch_size, max_num_nodes)

    target = torch.zeros(batch_size, max_num_nodes)
    edge_mask = torch.zeros(batch_size, max_num_nodes, max_num_nodes, dtype=torch.bool)
    q_mask = torch.zeros(batch_size, max_num_nodes, dtype=torch.bool)

    for i, (g, _, _) in enumerate(samples):

        eta[i, :batch_num_nodes[i]]      = g.nodes['high'].data['eta']
        phi[i, :batch_num_nodes[i]]      = g.nodes['high'].data['phi']
        cosphi[i, :batch_num_nodes[i]]   = g.nodes['high'].data['cosphi']
        sinphi[i, :batch_num_nodes[i]]   = g.nodes['high'].data['sinphi']
        layer[i, :batch_num_nodes[i]]    = g.nodes['high'].data['layer']

        e_truth[i, :batch_num_nodes[i]]  = g.nodes['high'].data['e_truth']
        e_proxy[i, :batch_num_nodes[i]]  = g.nodes['high'].data['e_proxy']

        eta_raw[i, :batch_num_nodes[i]]     = g.nodes['high'].data['eta_raw']
        e_truth_raw[i, :batch_num_nodes[i]] = g.nodes['high'].data['e_truth_raw']
        if 'e_proxy_raw' in g.nodes['high'].data:
            e_proxy_raw[i, :batch_num_nodes[i]] = g.nodes['high'].data['e_proxy_raw']

        target[i, :batch_num_nodes[i]] = g.nodes['high'].data['target']
        q_mask[i, :batch_num_nodes[i]] = True
        
        src, dst = g.edges(etype='high_to_high')
        edge_mask[i, src, dst] = True

    cond_trans_objs = [x[1] for x in samples]

    return_dict = {'layer': layer.unsqueeze(-1), 'eta': eta.unsqueeze(-1), 
            'phi': phi.unsqueeze(-1), 'cosphi': cosphi.unsqueeze(-1), 'sinphi': sinphi.unsqueeze(-1),
            'e_truth': e_truth.unsqueeze(-1), 'e_proxy': e_proxy.unsqueeze(-1), 'target': target.unsqueeze(-1),
            'eta_raw': eta_raw.unsqueeze(-1), 'e_truth_raw': e_truth_raw.unsqueeze(-1),
            'q_mask': q_mask.bool(), 'edge_mask': edge_mask.bool(),
            'cond_trans_obj': cond_trans_objs, 'idx': [x[2] for x in samples]}

    if 'e_proxy_raw' in samples[0][0].nodes['high'].data:
        return_dict['e_proxy_raw'] = e_proxy_raw.unsqueeze(-1)

    return return_dict


def collate_graphs_plus(samples):

    '''
        This is the one if we want the low cells as well
    '''

    ret_dict = collate_graphs(samples)

    batch_size = len(samples)
    batch_num_nodes = [x[0].number_of_nodes('low') for x in samples]
    max_num_nodes = max(batch_num_nodes)

    eta_raw = torch.zeros(batch_size, max_num_nodes)
    phi = torch.zeros(batch_size, max_num_nodes)
    sinphi = torch.zeros(batch_size, max_num_nodes)
    cosphi = torch.zeros(batch_size, max_num_nodes)
    layer = torch.zeros(batch_size, max_num_nodes, dtype=torch.int)
    e_meas_raw = torch.zeros(batch_size, max_num_nodes)

    q_mask = torch.zeros(batch_size, max_num_nodes, dtype=torch.bool)

    for i, (g, _, _) in enumerate(samples):
        eta_raw[i, :batch_num_nodes[i]]= g.nodes['low'].data['eta_raw']
        phi[i, :batch_num_nodes[i]]    = g.nodes['low'].data['phi']
        cosphi[i, :batch_num_nodes[i]] = g.nodes['low'].data['cosphi']
        sinphi[i, :batch_num_nodes[i]] = g.nodes['low'].data['sinphi']
        layer[i, :batch_num_nodes[i]]  = g.nodes['low'].data['layer']
        e_meas_raw[i, :batch_num_nodes[i]] = g.nodes['low'].data['e_meas_raw']
        q_mask[i, :batch_num_nodes[i]] = True

    ret_dict['low_eta_raw'] = eta_raw.unsqueeze(-1)
    ret_dict['low_phi'] = phi.unsqueeze(-1)
    ret_dict['low_cosphi'] = cosphi.unsqueeze(-1)
    ret_dict['low_sinphi'] = sinphi.unsqueeze(-1)
    ret_dict['low_layer'] = layer.unsqueeze(-1)
    ret_dict['low_e_meas_raw'] = e_meas_raw.unsqueeze(-1)
    ret_dict['low_q_mask'] = q_mask.bool()


    # particles (no zero padding)
    if 'pt' in samples[0][0].nodes['particle'].data:
        ret_dict['particle_pt'] = [x[0].nodes['particle'].data['pt'] for x in samples]
        ret_dict['particle_eta'] = [x[0].nodes['particle'].data['eta'] for x in samples]
        ret_dict['particle_phi'] = [x[0].nodes['particle'].data['phi'] for x in samples]
        ret_dict['particle_e'] = [x[0].nodes['particle'].data['e'] for x in samples]
        ret_dict['particle_pdgid'] = [x[0].nodes['particle'].data['pdgid'] for x in samples]
        ret_dict['particle_dep_e'] = [x[0].nodes['particle'].data['dep_e'] for x in samples]
    

    # incidence matrices (no zero padding)
    if 'e_part_0' in samples[0][0].nodes['high'].data:
        num_particles = samples[0][0].number_of_nodes('particle')
        for i in range(num_particles):
            ret_dict[f'high_e_part_{i}'] = [x[0].nodes['high'].data[f'e_part_{i}'] for x in samples]
            ret_dict[f'low_e_part_{i}'] = [x[0].nodes['low'].data[f'e_part_{i}'] for x in samples]

    return ret_dict



import uproot
import numpy as np
import awkward as ak
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from glob import glob

from utility.transformation import VarTransformation


class PflowDataset(Dataset):
    def __init__(self, glob_arg, config_mv, reduce_ds=-1, energy_threshold=0, 
            res='low', drop_single_part_events=False, load_incidence=False):

        self.config_mv = config_mv
        self.reduce_ds = reduce_ds
        self.energy_threshold = energy_threshold
        self.res = res
        self.load_incidence = load_incidence
        self.max_part = config_mv['pf_model']['max_particles']

        self.tree_name = 'Low_Tree'
        self.energy_branch_name = 'e_meas_raw'
        if res == 'high':
            self.tree_name = 'High_Tree'
            self.energy_branch_name = 'e_pred_raw'

        filepaths = glob(glob_arg)
        filepaths.sort( key=lambda x: int(x.split('_')[-2]) )
        self.load_data(filepaths)

        if drop_single_part_events: # useful when training only kinematics
            n_part_mask = np.where(np.array([len(x) > 1 for x in self.data_dict['particle_e']]))[0]
            for k, v in self.data_dict.items():
                self.data_dict[k] = np.array([v[i] for i in n_part_mask], dtype=object)

        self.n_events = len(self.data_dict['particle_e'])
        print(f'\n{len(self)} events loaded')

        self.cell_count = [len(x) for x in self.data_dict['cell_e']]
        print(f'max cell count: {max(self.cell_count)}')

        # create the transform dict
        self.transform_dicts = {}
        for k, v in self.config_mv['var_transform'].items():
            self.transform_dicts[k] = VarTransformation(v)

        self.pdgid_to_class = {
              -11:  1,   11:  1, # e
               22:  0, # photon
        }



    def load_data(self, filepaths):
        self.cell_vars = ['cell_e', 'cell_eta', 'cell_phi', 'cell_layer']
        if self.load_incidence:
            for pi in range(self.max_part):
                self.cell_vars.append(f'e_part_{pi}')

        self.particle_vars = [
            'particle_pt', 'particle_e', 'particle_eta', 'particle_phi', 'particle_pdgid', 'particle_dep_e']

        self.data_dict = {}
        for v in self.cell_vars + self.particle_vars:
            self.data_dict[v] = []

        self.n_events = 0

        for filepath in tqdm(filepaths, desc='Loading files', total=len(filepaths)):
            with uproot.open(filepath) as file:

                # reading cells
                tree = file[self.tree_name]

                e_stop = None
                if self.reduce_ds != -1 and self.n_events + tree.num_entries > self.reduce_ds:
                    e_stop = self.reduce_ds - self.n_events

                _energy = tree[self.energy_branch_name].array(entry_stop=e_stop)
                mask = _energy > self.energy_threshold # MeV cut

                energy  = [np.array(x) for x in _energy[mask]]
                eta_raw = [np.array(x) for x in tree['eta_raw'].array(entry_stop=e_stop)[mask]]
                phi     = [np.array(x) for x in tree['phi'].array(entry_stop=e_stop)[mask]]
                layer   = [np.array(x) for x in tree['layer'].array(entry_stop=e_stop)[mask]]

                self.data_dict['cell_e'].extend(energy)
                self.data_dict['cell_eta'].extend(eta_raw)
                self.data_dict['cell_phi'].extend(phi)
                self.data_dict['cell_layer'].extend(layer)

                if self.load_incidence:
                    for pi in range(self.max_part):
                        self.data_dict[f'e_part_{pi}'].extend(
                            [np.array(x) for x in tree[f'e_part_{pi}'].array(entry_stop=e_stop)[mask]])

                # reading particles
                tree = file['Particle_Tree']

                self.data_dict['particle_pt'].extend([
                    np.array(x) for x in tree['particle_pt'].array(entry_stop=e_stop)])
                self.data_dict['particle_e'].extend([
                    np.array(x) for x in tree['particle_e'].array(entry_stop=e_stop)])
                self.data_dict['particle_eta'].extend([
                    np.array(x) for x in tree['particle_eta'].array(entry_stop=e_stop)])
                self.data_dict['particle_phi'].extend([
                    np.array(x) for x in tree['particle_phi'].array(entry_stop=e_stop)])
                self.data_dict['particle_pdgid'].extend([
                    np.array(x) for x in tree['particle_pdgid'].array(entry_stop=e_stop)])
                self.data_dict['particle_dep_e'].extend([
                    np.array(x) for x in tree['particle_dep_e'].array(entry_stop=e_stop)])

                self.n_events += len(energy)

            if self.reduce_ds != -1 and self.n_events >= self.reduce_ds:
                break


    def __len__(self):
        return self.n_events


    def __getitem__(self, idx):

        # load data
        cell_data = {}
        cell_data['phi']     = torch.from_numpy(self.data_dict['cell_phi'][idx])
        cell_data['eta_raw'] = torch.from_numpy(self.data_dict['cell_eta'][idx])
        cell_data['e_raw']   = torch.from_numpy(self.data_dict['cell_e'][idx])
        cell_data['layer']   = torch.from_numpy(self.data_dict['cell_layer'][idx])

        particle_data = {}
        particle_data['e_raw']     = torch.from_numpy(self.data_dict['particle_e'][idx])
        particle_data['pt_raw']    = torch.from_numpy(self.data_dict['particle_pt'][idx])
        particle_data['eta_raw']   = torch.from_numpy(self.data_dict['particle_eta'][idx])
        particle_data['phi']       = torch.from_numpy(self.data_dict['particle_phi'][idx])
        particle_data['dep_e_raw'] = torch.from_numpy(self.data_dict['particle_dep_e'][idx])

        # cos and sin
        cell_data['cosphi'] = torch.cos(cell_data['phi'])
        cell_data['sinphi'] = torch.sin(cell_data['phi'])

        # scale
        cell_data['e']   = self.transform_dicts['e'].forward(cell_data['e_raw'])
        cell_data['eta'] = self.transform_dicts['eta'].forward(cell_data['eta_raw'])
        
        # particles
        particle_data['pt']  = self.transform_dicts['pt'].forward(particle_data['pt_raw'])
        particle_data['e']   = self.transform_dicts['e'].forward(particle_data['e_raw'])
        particle_data['eta'] = self.transform_dicts['eta'].forward(particle_data['eta_raw'])
        particle_data['dep_e'] = self.transform_dicts['e'].forward(particle_data['dep_e_raw'])

        # pdgid to class
        particle_data['particle_class'] = torch.LongTensor(
            [self.pdgid_to_class[x] for x in self.data_dict['particle_pdgid'][idx]])

        # cardinality
        n_particles = particle_data['e_raw'].shape[0]   

        return_tuple = (cell_data, n_particles, particle_data, idx)

        # incidence
        if self.load_incidence:
            energy_matrix = torch.zeros(len(cell_data['e_raw']), self.max_part)
            for pi in range(self.max_part):
                energy_matrix[:, pi] = torch.from_numpy(self.data_dict[f'e_part_{pi}'][idx])
            energy_matrix_row_sum = energy_matrix.sum(dim=1, keepdim=True)
            energy_matrix_row_sum[energy_matrix_row_sum == 0] = 1
            incidence_matrix = energy_matrix / energy_matrix_row_sum

            return_tuple += (incidence_matrix,)

        return return_tuple


    
def collate_fn(samples, max_part=None):

    batch_size = len(samples)
    
    # cells
    batch_num_cells = [len(x[0]['e_raw']) for x in samples]
    max_num_cells = max(batch_num_cells)

    cell_e      = torch.zeros(batch_size, max_num_cells)
    cell_eta    = torch.zeros(batch_size, max_num_cells)
    cell_phi    = torch.zeros(batch_size, max_num_cells)
    cell_cosphi = torch.zeros(batch_size, max_num_cells)
    cell_sinphi = torch.zeros(batch_size, max_num_cells)
    cell_layer  = torch.zeros(batch_size, max_num_cells, dtype=torch.int)

    cell_e_raw   = torch.zeros(batch_size, max_num_cells)
    cell_eta_raw = torch.zeros(batch_size, max_num_cells)

    part_pt  = torch.zeros(batch_size, max_part)
    part_e   = torch.zeros(batch_size, max_part)
    part_eta = torch.zeros(batch_size, max_part)
    part_phi = torch.zeros(batch_size, max_part)
    part_dep_e = torch.zeros(batch_size, max_part)
    part_class = torch.zeros(batch_size, max_part, dtype=torch.int)

    part_pt_raw  = torch.zeros(batch_size, max_part)
    part_e_raw   = torch.zeros(batch_size, max_part)
    part_eta_raw = torch.zeros(batch_size, max_part)
    part_dep_e_raw = torch.zeros(batch_size, max_part)


    # masks (1: real , 0: padding) Need to flip them for transformer
    cell_mask = torch.zeros(batch_size, max_num_cells).bool()
    part_mask = torch.zeros(batch_size, max_part).bool()

    for i, sample in enumerate(samples):

        c_dict, n_part, p_dict = sample[:3]

        cell_e[i, :batch_num_cells[i]]      = c_dict['e']
        cell_eta[i, :batch_num_cells[i]]    = c_dict['eta']
        cell_phi[i, :batch_num_cells[i]]    = c_dict['phi']
        cell_cosphi[i, :batch_num_cells[i]] = c_dict['cosphi']
        cell_sinphi[i, :batch_num_cells[i]] = c_dict['sinphi']
        cell_layer[i, :batch_num_cells[i]]  = c_dict['layer']

        cell_e_raw[i, :batch_num_cells[i]]   = c_dict['e_raw']
        cell_eta_raw[i, :batch_num_cells[i]] = c_dict['eta_raw']

        part_pt[i, :n_part]  = p_dict['pt']
        part_e[i, :n_part]   = p_dict['e']
        part_eta[i, :n_part] = p_dict['eta']
        part_phi[i, :n_part] = p_dict['phi']
        part_dep_e[i, :n_part] = p_dict['dep_e']
        part_class[i, :n_part] = p_dict['particle_class']

        part_pt_raw[i, :n_part]  = p_dict['pt_raw']
        part_e_raw[i, :n_part]   = p_dict['e_raw']
        part_eta_raw[i, :n_part] = p_dict['eta_raw']
        part_dep_e_raw[i, :n_part] = p_dict['dep_e_raw']

        cell_mask[i, :batch_num_cells[i]] = True
        part_mask[i, :n_part] = True

    # particle cardinality
    cardinality = torch.LongTensor([x[1] for x in samples])

    batch_dict =  {
        'cell_e': cell_e, 'cell_eta': cell_eta, 'cell_phi': cell_phi,
        'cell_cosphi': cell_cosphi, 'cell_sinphi': cell_sinphi,
        'cell_layer': cell_layer, 'cell_mask': cell_mask,
        'cell_e_raw': cell_e_raw, 'cell_eta_raw': cell_eta_raw,

        'part_pt': part_pt, 'part_e': part_e, 'part_dep_e': part_dep_e,
        'part_eta': part_eta, 'part_phi': part_phi,
        'part_class': part_class, 'part_mask': part_mask,
        'part_pt_raw': part_pt_raw, 'part_e_raw': part_e_raw, 'part_dep_e_raw': part_dep_e_raw,
        'part_eta_raw': part_eta_raw,
        
        'cardinality': cardinality, 'idx': torch.LongTensor([x[3] for x in samples])
    }

    if len(samples[0]) == 5:
        inc_mat_batched = torch.zeros(batch_size, max_num_cells, max_part)
        for i, (_, _, _, _, inc_mat) in enumerate(samples):
            inc_mat_batched[i, :batch_num_cells[i], :inc_mat.shape[1]] = inc_mat
        batch_dict['incidence_matrix'] = inc_mat_batched

    return batch_dict

import uproot
import sys
import yaml
import numpy as np
import argparse
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, help='Path to config file', required=True)
args = parser.parse_args()

config_path = args.config
with open(config_path, 'r') as fp:
    config = yaml.safe_load(fp)


filepath = config['train_path']
tree = uproot.open(filepath)['High_Tree']
n_events = tree.num_entries
print("Number of events: ", n_events)

cell_x = np.hstack(tree['cell_x'].array(library='np'))
cell_y = np.hstack(tree['cell_y'].array(library='np'))
cell_z = np.hstack(tree['cell_z'].array(library='np'))

def custom_print(var, x, trans, scale_mode, m=None):
    print(f'"{var}": {{')

    if trans == "null":
        print(f'    "transformation": {trans},')
    else:
        print(f'    "transformation": "{trans}",')
        
    if m is not None:
        print(f'    "m": {m},')

    print(f'    "scale_mode": "{scale_mode}",')    
    print(f'    "mean": {np.mean(x):.3f}, "std": {np.std(x):.3f},')
    print(f'    "min": {np.min(x):.3f}, "max": {np.max(x):.3f}, "range": [-1,1]')
    print('},')



custom_print('x', cell_x, trans='null', scale_mode='standard')
custom_print('y', cell_y, trans='null', scale_mode='standard')
custom_print('z', cell_z, trans='null', scale_mode='standard')
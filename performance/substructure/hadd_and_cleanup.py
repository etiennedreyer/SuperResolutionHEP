import os
import uproot
from pathlib import Path
import numpy as np
import argparse
import glob

# argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--substructure_dir', '-dir', type=str, required=True)
args = argparser.parse_args()

substructure_dir = args.substructure_dir


print('checks -')

# check if all the statuses are removed
status_files = glob.glob(f'{substructure_dir}/status/job_*.status')
if len(status_files) > 0:
    raise ValueError('status files found! (jobs are still running)')
print('\tstatus files are removed! (all jobs finished)')


# read all the error files and make sure they are all empty
error_files = glob.glob(f'{substructure_dir}/error_*.log')
for ef in error_files:
    with open(ef, 'r') as f:
        content = f.read()
        if len(content) > 0:
            raise ValueError(f'error file {ef} is not empty!')
print('\terror files are empty!')
        

# read all npz files and concatenate them
npz_files = glob.glob(f'{substructure_dir}/substructures_*.npz')
npz_files = sorted(npz_files, key=lambda x: int(x.split('_')[-2]))

d2_low, c2_low, c3_low = [], [], []
d2_low_split, c2_low_split, c3_low_split = [], [], []
d2_high_truth, c2_high_truth, c3_high_truth = [], [], []
d2_high_pred, c2_high_pred, c3_high_pred = [], [], []

for npz_file in npz_files:
    with np.load(npz_file) as f:
        d2_low.append(f['d2_low'])
        c2_low.append(f['c2_low'])
        c3_low.append(f['c3_low'])

        d2_low_split.append(f['d2_low_split'])
        c2_low_split.append(f['c2_low_split'])
        c3_low_split.append(f['c3_low_split'])

        d2_high_truth.append(f['d2_high_truth'])
        c2_high_truth.append(f['c2_high_truth'])
        c3_high_truth.append(f['c3_high_truth'])

        d2_high_pred.append(f['d2_high_pred'])
        c2_high_pred.append(f['c2_high_pred'])
        c3_high_pred.append(f['c3_high_pred'])


d2_low = np.hstack(d2_low)
c2_low = np.hstack(c2_low)
c3_low = np.hstack(c3_low)

d2_low_split = np.hstack(d2_low_split)
c2_low_split = np.hstack(c2_low_split)
c3_low_split = np.hstack(c3_low_split)

d2_high_truth = np.hstack(d2_high_truth)
c2_high_truth = np.hstack(c2_high_truth)
c3_high_truth = np.hstack(c3_high_truth)

d2_high_pred = np.hstack(d2_high_pred)
c2_high_pred = np.hstack(c2_high_pred)
c3_high_pred = np.hstack(c3_high_pred)


# write them into uproot tree
save_path = os.path.join(substructure_dir, 'substructures.root')
with uproot.recreate(save_path) as f:
    f['substructures'] = {
        'd2_low': d2_low,
        'c2_low': c2_low,
        'c3_low': c3_low,
        
        'd2_low_split': d2_low_split,
        'c2_low_split': c2_low_split,
        'c3_low_split': c3_low_split,

        'd2_high_truth': d2_high_truth,
        'c2_high_truth': c2_high_truth,
        'c3_high_truth': c3_high_truth,
        
        'd2_high_pred': d2_high_pred,
        'c2_high_pred': c2_high_pred,
        'c3_high_pred': c3_high_pred
    }
print(f'Substructures are saved to {save_path}')

answer = input('\ncleanup? (y/n): ')
if answer == 'y':
    for npz_file in npz_files:
        os.remove(npz_file)
    print('\tnpz files are removed!')

    for ef in error_files:
        os.remove(ef)
    print('\terror files are removed!')
    
    output_files = glob.glob(f'{substructure_dir}/output_*.log')
    for of in output_files:
        os.remove(of)
    print('\toutput files are removed!')

    status_dir = os.path.join(substructure_dir, 'status')
    os.rmdir(status_dir)
    print('\tstatus dir is removed!')
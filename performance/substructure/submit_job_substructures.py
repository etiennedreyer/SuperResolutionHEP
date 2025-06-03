import os
import uproot
from pathlib import Path
import numpy as np
import argparse


ncpus = '1'
mem   = '3gb'
walltime = '11:00:00'
run_dir = os.path.dirname(os.path.realpath(__file__))


# argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--file_path', '-fp', type=str, required=True)
argparser.add_argument('--ncpus', '-nc', type=str, required=False, default=ncpus)
argparser.add_argument('--mem', '-mem', type=str, required=False, default=mem)
argparser.add_argument('--chunk_size', '-cs', type=int, required=False, default=10)
args = argparser.parse_args()

file_path = args.file_path
ncpus = args.ncpus
mem = args.mem
chunk_size = args.chunk_size



# read the file
with uproot.open(file_path) as f:
    tree = f['Low_Tree']
    n_events = tree.num_entries

indices = np.arange(n_events)
splitted_indices = np.array_split(indices, np.ceil(n_events / args.chunk_size))

print('total job count:', len(splitted_indices), '\n')

dst_dir = Path(file_path).parent
dst_dir = os.path.join(dst_dir, 'substructures')
os.makedirs(dst_dir, exist_ok=True)

status_dir = os.path.join(dst_dir, 'status')
os.makedirs(status_dir, exist_ok=True)


for i, sp_idxs in enumerate(splitted_indices):
    entry_start = sp_idxs[0]
    entry_stop  = sp_idxs[-1] + 1

    # create a tmp file to keep track of the status (no need to write anything in it)
    tmp_file = os.path.join(status_dir, f'job_{entry_start}_{entry_stop}.status')
    with open(tmp_file, 'w') as f:
        pass


    command  = f'qsub -o {dst_dir}/output_{i}.log'
    command += f' -e {dst_dir}/error_{i}.log'
    command += f' -q N -N sr_substr -l walltime={walltime},mem={args.mem},ncpus={args.ncpus},io=1'
    command += f' -v FILE_PATH="{file_path}",ENTRY_START="{entry_start}",ENTRY_STOP="{entry_stop}",RUN_DIR="{run_dir}",SAVE_DIR="{dst_dir}"'
    command += f' {run_dir}/run_on_node_substructures.sh'

    print(f'submitting job {i}/{len(splitted_indices)-1}...')
    os.system(command)

import comet_ml
import glob

import sys
paths = sys.path
for p in paths:
     if '.local' in p:
             paths.remove(p)

import os
import yaml
import argparse
import torch

from lightning import SupResLightning
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer


import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


if __name__ == "__main__":

    os.system('nvidia-smi')

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_path_model_and_var', '-cmv', type=str, required=True)
    argparser.add_argument('--config_path_train', '-ct', type=str, required=True)
    argparser.add_argument('--exp_key', '-ekey', type=str, required=False)
    argparser.add_argument('--debug_mode', '-d', action='store_true')
    argparser.add_argument('--precision', '-p', type=str, required=False, default='highest')
    argparser.add_argument('--gpu', '-g', type=str, required=False, default='0')
    args = argparser.parse_args()
    
    config_path_mv = args.config_path_model_and_var
    config_path_t = args.config_path_train
    debug_mode = args.debug_mode
    exp_key = args.exp_key
    precision = args.precision

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_float32_matmul_precision(precision)


    with open(config_path_mv, 'r') as fp:
        config_mv = yaml.safe_load(fp)

    with open(config_path_t, 'r') as fp:
        config_t = yaml.safe_load(fp)


    net = SupResLightning(config_mv, config_t)

    replace_sampler_ddp = True
    num_dev = -1
    strategy = None

    # for saving checkpoints for best 3 models (according to val loss) and last epoch
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss_raw',
        mode='min',
        every_n_train_steps=0,
        every_n_epochs=1,
        train_time_interval=None,
        save_top_k=3,
        save_last= True,
        filename='{epoch}-{val/loss:.4f}')


    if debug_mode:
        print('\033[96m' + 'Running in debug mode' + '\033[0m')
        trainer = Trainer(
            max_epochs = config_t['num_epochs'],
            accelerator = config_t['device'],
            devices = num_dev,
            default_root_dir = config_t["base_root_dir"],
            strategy = strategy,
            replace_sampler_ddp = replace_sampler_ddp,
            callbacks = [checkpoint_callback],
            check_val_every_n_epoch = config_t['eval_every_n_epoch'],
            # gradient_clip_val=1.0
        )

    else:
        comet_logger = CometLogger(
            api_key=os.environ["COMET_API_KEY"],
            project_name=config_t["project_name"],
            workspace=os.environ["COMET_WORKSPACE"],
            experiment_name=config_t["run_name"],
            experiment_key=exp_key
        )

        net.set_comet_logger(comet_logger)
        comet_logger.experiment.log_asset(config_path_mv, file_name='config_mv')
        comet_logger.experiment.log_asset(config_path_t, file_name='config_t')

        # log files
        dirs2log = ['.', 'models', 'utility']
        for d in dirs2log:
            all_files = glob.glob(f'{d}/*.py')
            for fpath in all_files:
                comet_logger.experiment.log_asset(fpath, file_name=f'{d}/{fpath}')

        trainer = Trainer(
            max_epochs = config_t['num_epochs'],
            accelerator = config_t['device'],
            devices = num_dev,
            default_root_dir = config_t["base_root_dir"],
            strategy = strategy,
            replace_sampler_ddp = replace_sampler_ddp,
            callbacks = [checkpoint_callback],
            check_val_every_n_epoch = config_t['eval_every_n_epoch'],
            log_every_n_steps = 1,
            logger = comet_logger,
            # gradient_clip_val=1.0
        )
    
    trainer.fit(net, ckpt_path=config_t['resume_from_checkpoint'])
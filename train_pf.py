import comet_ml
import glob

import sys, os
paths = sys.path
for p in paths:
     if '.local' in p:
             paths.remove(p)

import yaml
import argparse
import torch

from pflow.lightning_pf import PflowLightning
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import os

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
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_float32_matmul_precision(args.precision)

    config_path_mv = args.config_path_model_and_var
    with open(config_path_mv, 'r') as fp:
        config_mv = yaml.safe_load(fp)

    config_path_t = args.config_path_train
    with open(config_path_t, 'r') as fp:
        config_t = yaml.safe_load(fp)


    lightning_model = PflowLightning(config_mv, config_t)


    # for saving checkpoints for best 3 models (according to val loss) and last epoch
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss_to_optimize_on',
        mode='min',
        every_n_train_steps=0,
        every_n_epochs=1,
        train_time_interval=None,
        save_top_k=3,
        save_last= True,
        filename='{epoch}-{val_loss_to_optimize_on:.4f}')


    # debug mode
    if args.debug_mode:
        print('\033[96m' + 'Running in debug mode' + '\033[0m')
        trainer = Trainer(
            max_epochs = config_t['num_epochs'],
            accelerator = config_t['device'],
            default_root_dir = config_t["base_root_dir"],
            callbacks = [checkpoint_callback],
            check_val_every_n_epoch = config_t['eval_every_n_epoch'],
        )

    # normal mode
    else:
        comet_logger = CometLogger(
            api_key = os.environ["COMET_API_KEY"],
            project_name = config_t["project_name"],
            workspace = os.environ["COMET_WORKSPACE"],
            experiment_name = config_t["run_name"],
            experiment_key = args.exp_key
        )

        lightning_model.set_comet_logger(comet_logger)
        comet_logger.experiment.log_asset(config_path_mv, file_name='configs/config_mv')
        comet_logger.experiment.log_asset(config_path_t, file_name='configs/config_t')

        # log files
        dirs2log = ['.', 'models', 'utility', 'pflow', 'pflow/models']
        for d in dirs2log:
            all_files = glob.glob(f'{d}/*.py')
            for fpath in all_files:
                file_name_in_comet = os.path.normpath(os.path.join('code', fpath))
                comet_logger.experiment.log_asset(fpath, file_name=file_name_in_comet)

        trainer = Trainer(
            max_epochs = config_t['num_epochs'],
            accelerator = config_t['device'],
            devices = 1,
            default_root_dir = config_t['base_root_dir'],
            logger = comet_logger,
            callbacks = [checkpoint_callback],
            check_val_every_n_epoch = config_t['eval_every_n_epoch'],
            log_every_n_steps = 1,
            # gradient_clip_val=1.0,
            # gradient_clip_algorithm='norm',
        )
    
    trainer.fit(lightning_model, ckpt_path=config_t['resume_from_checkpoint'])

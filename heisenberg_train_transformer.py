"""
script to train conditional transformers for Heisenberg data
"""
import argparse
from datetime import datetime as dt
import json, random
import numpy as np
from operator import itemgetter
import os
import shutil
import sys
import torch
import pandas as pd

from constants import *
from src.data.loading import DatasetGCTransformer
from src.training import GCTransformerTrainer
from src.models.graph_encoder import get_graph_encoder
from src.models.gctransformer import init_gctransformer, get_sample_structure
from src.utils import timestamp, filter_folder_for_files
from src.training.utils import dir_setup, Logger

from src.models.mlp import MLP
from src.models.transformer import init_conditional_transformer
from src.training.rydberg_trainers import RydbergConditionalTransformerTrainer

from read_data import load_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='./saved_models')
    parser.add_argument('--data_dir', type=str,
                        default='N_10_NM_1000')
    parser.add_argument('--train-size', type=str, default=None)
    parser.add_argument('--tf-arch', type=str, default='transformer_l4_d128_h4')
    parser.add_argument('--gcn-arch', type=str, default='gcn_proj_3_16')
    parser.add_argument('--gcn-features', type=str, default=ONE_HOT_FEATURE,
                        choices=[WDEGREE_FEATURE, ONE_HOT_FEATURE])
    parser.add_argument('--hamiltonians', type=int, default=None,
                        help='number of training hamiltonians; set to None to use all '
                             'hamiltonians in the train split')
    parser.add_argument('--train-samples', type=int, default=1000,
                        help='number of train samples per hamiltonian')
    parser.add_argument('--iterations', type=int, default=20000,
                        help='number of epochs if epoch_mode = 1, else number of steps')
    parser.add_argument('--eval-every', type=int, default=1)
    parser.add_argument('--eval-test', type=int, default=1, choices=[0, 1])
    parser.add_argument('--k', type=int, default=1,
                        help='number of buckets for median of means estimation')
    parser.add_argument('--n_cpu', type=int, default=8,
                        help='number of cpu threads to use during batch generation')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--epoch_mode', type=int, default=0, choices=[0, 1])
    parser.add_argument('--sample-structure', type=int, default=2, choices=[0, 1, 2])

    parser.add_argument('--n_vars', type=int, default=9)
    parser.add_argument('--num_measurements', type=int, default=1000)
    parser.add_argument('--num_qubits', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=5000)
    args = parser.parse_args()
    return args


def get_hyperparams(**kwargs):
    hparams = argparse.Namespace(
        lr=1e-3,
        wd=0,
        bs=512,
        dropout=0.0,
        lr_scheduler=WARMUP_COSINE_SCHEDULER,
        warmup_frac=0.,
        final_lr=1e-7,
        smoothing=0.0,
        use_padding=0,
        val_frac=0.25,
        cattn=0
    )

    for k, v in kwargs.items():
        setattr(hparams, k, v)

    return hparams


def train_transformer(args, hparams):
    if args.debug:
        args.data_root = './data/conditional_heisenberg/'
        args.results_dir = './results-debug-local'
        args.train_samples = 20
        args.iterations = 2
        hparams.bs = 20

    # convert strings to integers
    # rows, cols = tuple(map(int, args.train_size.split('x')))

    # setup results dir structure
    # model_id = f'{args.gcn_arch}-{args.tf_arch}_feat{args.gcn_features}'

    # train id based on hyperparams
    train_id = f'iter{args.iterations}_lr{hparams.lr}_wd{hparams.wd}_bs{hparams.bs}_dropout{hparams.dropout}'
    train_id = train_id + f'_lrschedule{hparams.lr_scheduler}_'
    train_id = train_id + dt.now().strftime('%Y%m%d-%H%M%S')

    # dataset id
    dataset_id = f'N{args.num_qubits}'

    results_dir = os.path.join(args.results_dir, f"{dataset_id}_{train_id}")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # save terminal output to file
    sys.stdout = Logger(print_fp=os.path.join(results_dir, 'train_out.txt'))

    # save args
    with open(os.path.join(results_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    # save hparams
    with open(os.path.join(results_dir, 'hparams.json'), 'w') as f:
        json.dump(vars(hparams), f)

    if args.verbose:
        print_dict(vars(args))
        print_dict(vars(hparams))

    # generator for random numbers
    # rng = np.random.default_rng(seed=args.seed)
    set_seed(args.seed)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    d_model = TF_ARCHS[args.tf_arch]['d_model']
    n_head = TF_ARCHS[args.tf_arch]['n_head']
    n_layers = TF_ARCHS[args.tf_arch]['n_layers']


    assert d_model % n_head == 0, 'd_model must be integer multiple of n_head!'

    # initialize graph encoder
    # qubits = rows * cols
    # in_node_dim = 1 if args.gcn_features == WDEGREE_FEATURE else qubits
    encoder = MLP(input_size=args.n_vars, output_size=d_model, 
                  n_layers=1, hidden_size=128, activation='ELU', 
                  input_layer_norm=False,
                  output_batch_size=None, device=device,
                  output_factor=1.)

    # initialize transformer
    transformer = init_conditional_transformer(
        n_outcomes=6,
        encoder=encoder,
        n_layers=n_layers,
        d_model=d_model,
        d_ff=4 * d_model,
        n_heads=n_head,
        dropout=hparams.dropout,
        version=hparams.use_padding,
        use_prompt=False, #***
)

    transformer.to(device)

    train_set = load_data(args, split='train')
    rng = np.random.default_rng(seed=args.seed)

    print(f'[{timestamp()}] start training, saving results to {results_dir}')
    trainer = RydbergConditionalTransformerTrainer(model=transformer,
                                  train_dataset=train_set,
                                  test_dataset=None,
                                  iterations=args.iterations,
                                  lr=hparams.lr,
                                  final_lr=hparams.final_lr,
                                  lr_scheduler=hparams.lr_scheduler,
                                  warmup_frac=hparams.warmup_frac,
                                  weight_decay=hparams.wd,
                                  batch_size=hparams.bs,
                                  rng=rng,
                                  save_interval=args.save_interval,
                                  save_dir=results_dir,
                                  smoothing=hparams.smoothing,
                                  eval_every=args.eval_every,
                                  transfomer_version=hparams.use_padding,
                                  device=device)

    trainer.train()

    # pstr = f'[{timestamp()}] training end, train total-loss: {train_total_loss:.4f}'
    # pstr = pstr + f', test total-loss: {test_total_loss:.4f}, val total-loss: {val_total_loss:.4f}'
    # print(pstr)

    # trainer.save_model('final', is_best=False)


def print_dict(d, tag=None):
    """ helper function to print args """
    print(f'--------{tag or ""}----------')
    for k, v in d.items():
        print('{0:27}: {1}'.format(k, v))
    print(f'--------{tag or ""}----------\n')


def save_data(results_data_dir, data_dir, hamiltonian_ids):
    os.makedirs(results_data_dir)

    for fn in os.listdir(data_dir):
        _, ext = os.path.splitext(fn)

        if ext == '.json':
            shutil.copy(os.path.join(data_dir, fn), os.path.join(results_data_dir, fn))
            continue

        if ext not in ['.txt', '.npy']:
            continue

        hid = int(fn[(fn.find('id') + 2):fn.find(ext)])
        if hamiltonian_ids is None or hid in hamiltonian_ids:
            shutil.copy(os.path.join(data_dir, fn), os.path.join(results_data_dir, fn))

    
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available


if __name__ == '__main__':
    train_transformer(args=parse_args(), hparams=get_hyperparams())

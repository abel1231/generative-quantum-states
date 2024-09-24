"""
script to sample from conditional transformers for Heisenberg data
"""

import argparse
import glob
import json, random
import numpy as np
import os
import torch

from constants import *
from src.models.gctransformer.generate_samples import generate_samples
from src.models.gctransformer import init_gctransformer, get_sample_structure
from src.models.graph_encoder import get_graph_encoder
from src.utils import filter_folder_for_files

from src.models.mlp import MLP
from src.models.transformer import init_conditional_transformer
from src.training.rydberg_trainers import RydbergConditionalTransformerTrainer
from heisenberg_train_transformer import load_data


def main():
    model_dir = 'saved_models/N10_iter20000_lr0.001_wd0_bs512_dropout0.0_lrschedulewarmup_cosine_20240924-002927'
    model_args, hparams = load_model_args(model_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default='N_10_NM_1000_S_20')
    parser.add_argument('--snapshots', type=int, default=20000,
                        help='number of generated samples for evaluation')
    # parser.add_argument('--num_measurements', type=int, default=1000)
    # parser.add_argument('--num_qubits', type=int, default=10)
    parser.add_argument('--seed2', type=int, default=123)
    add_dict_to_argparser(parser, model_args) # update latest args according to argparse
    args = parser.parse_args()
    args.data_dir = args.test_dir

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # generator for random numbers
    # rng = np.random.default_rng(seed=args.seed2)
    set_seed(args.seed2)

    data = load_data(args, split='test')
    
    for lst in glob.glob(model_dir):
        print(lst)
        checkpoints = sorted(glob.glob(f"{lst}/checkpoint*.pth.tar"))[::-1]

        out_dir = './generation_outputs'
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        for checkpoint_one in checkpoints:
            # intialize model
            ckpt_fp = checkpoint_one
            transformer = load_model(
                            ckpt_fp,
                            args,
                            hparams,
                            device=device
                        )

            save_dir = os.path.join(out_dir, checkpoint_one.split('/')[1])
            run_sampling(transformer, data, args, save_dir, args.snapshots, checkpoint_one.strip().split('/')[-1].split('.')[0])


def load_model_args(results_dir):
    # get and parse model args
    with open(os.path.join(results_dir, 'args.json'), 'r') as f:
        model_args = json.load(f)

    # get and parse hyperparams
    with open(os.path.join(results_dir, 'hparams.json'), 'r') as f:
        hparams = json.load(f)

    return model_args, hparams


def load_model(checkpoint_fp, args, hparams, device):
    d_model = TF_ARCHS[args.tf_arch]['d_model']
    n_head = TF_ARCHS[args.tf_arch]['n_head']
    n_layers = TF_ARCHS[args.tf_arch]['n_layers']
    # initialize graph encoder
    encoder = MLP(input_size=args.n_vars, output_size=d_model, 
                  n_layers=1, hidden_size=128, activation='ELU', 
                  input_layer_norm=False,
                  output_batch_size=None, device=device,
                  output_factor=1.)

    # structure of mmt sequence
    # sample_struct = get_sample_structure(version=sample_structure)

    # initialize transformer
    transformer = init_conditional_transformer(
        n_outcomes=6,
        encoder=encoder,
        n_layers=n_layers,
        d_model=d_model,
        d_ff=4 * d_model,
        n_heads=n_head,
        dropout=hparams['dropout'],
        version=hparams['use_padding'],
        use_prompt=False, #***
    )

    # load weights
    ckpt = torch.load(checkpoint_fp, map_location=device)

    transformer.load_state_dict(ckpt['model_state_dict'], strict=True)
    transformer.to(device)
    transformer.eval()
    print(f'loaded weights from {checkpoint_fp}')

    return transformer


def run_sampling(transformer, data, args, save_dir, snapshots, checkpoint_one):
    # generate samples for test hamiltonians
    test_data = data
    # couplings, couplings_ids = load_couplings(test_data_dir)
    # test_save_dir = os.path.join(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results = []
    for i, condition in enumerate(test_data['conditions']):
        condition = torch.from_numpy(np.array([condition])).float().to(transformer.device)
        gen_samples = transformer.sample_batch(cond_var=condition, batch_size=snapshots,
                                               num_qubits=args.num_qubits)

        # save samples
        gen_samples = gen_samples.astype(int)
        results.append(gen_samples)

    results = np.stack(results, axis=0)
    np.savetxt(os.path.join(save_dir, f'{checkpoint_one}.txt'), 
                gen_samples, 
                fmt='%d', delimiter=',')


def load_couplings(data_dir):
    # get data filepaths
    coupling_mats_fps = filter_folder_for_files(data_dir,
                                                file_pattern=f'coupling_matrix_id*.npy')
    coupling_mats_ids = [int(fp[(fp.find('id') + 2):fp.find('.npy')]) for fp in
                         coupling_mats_fps]

    coupling_matrices_array = []
    coupling_matrices_ids_list = []

    for cid, cfp in zip(coupling_mats_ids, coupling_mats_fps):
        # load coupling matrices
        coup_mat = np.load(os.path.join(data_dir, f'coupling_matrix_id{cid}.npy'))
        coupling_matrices_array.append(coup_mat)
        coupling_matrices_ids_list.append(cid)

    coupling_matrices_array = np.stack(coupling_matrices_array, axis=0)

    return coupling_matrices_array, coupling_matrices_ids_list


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
    

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available


if __name__ == '__main__':
    main()

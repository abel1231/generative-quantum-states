## Requirements

### Software

+ Python (tested with python 3.8 & 3.9)
    + Please see `requirements.txt` for the required Python packages (if you are using pip or Conda, you can
      run `pip install -r requirements.txt`)
    + Pytorch-Geometric (PyG): Please refer to
      its [official documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for
      installation.
        + PyG is used in our experiments on 2D Heisenberg models (our conditional generative model has a GNN module in
          that case).
    + Jax: Please refer to its [official documentation](https://github.com/google/jax#installation) for installation (a
      CPU version is sufficient).
        + `jax` and `neural-tangents` (a Jax-based package) are only used when comparing our method with Neural Tangent
          Kernel (used by Hsin-Yuan Huang et al. in their [Science 2022](https://arxiv.org/abs/2106.12627) paper). You
          can skip this installation if you do not intend to do the comparison.
+ Julia (tested with version 1.7 & 1.8)
    + The Julia language can be downloaded from the [official website](https://julialang.org/downloads/) (If you are
      using pip or Conda as your python package manager, you can also install Julia with
      the [Jill](https://github.com/johnnychen94/jill.py) package).
    + Run `julia install_pacakges.jl` in the `rydberg/` folder to install all necessary Julia packages.
        + If you want to use GPU for simulation, you need to install the `CUDA` and `Adapt` packages in addition (
          see `rydberg/README.md` for details).


### Pre-trained Models

We trained conditional transformers over the simulation data. If you want to directly load pre-trained models when using
our scripts, please put them in `logs/`.

#### Heisenberg Models

In Sec. III-A of the paper, we conducted various numerical experiments over Heisenberg models of various sizes.
The trained models are saved in `logs/2d_heisenberg_checkpoints/`.

#### Rydberg Atoms

In Sec. III-B of the paper, we conducted 4 machine learning experiments over Rydberg atom systems, and we provide
pre-trained models for them in different sub-folders. Notice that we trained models for a different number of
iterations, and marked them with suffixes like `...{#iterations}.pth` (e.g., `...100k.pth` implies training with 100k
iterations).

+ Sec. III-B(1) - _Predicting quantum phases_
    + `logs/rydberg_1D/`: models trained on 1D Rydberg-atom chains of 31 atoms.
    + `logs/rydberg_2D/`: models trained on 2D Rydberg-atom chains of 25 atoms (prepared with adiabatic evolution of 3.0
      μs)
+ Sec. III-B(2) - _Predicting phases of larger quantum systems_
    + `logs/rydberg_1D-size/`: models trained on 1D Rydberg-atom chains of 13,15,...,27 atoms.
+ Sec. III-B(3) - _Predicting phases of ground states prepared with longer adiabatic evolution time_
    + `logs/rydberg_2D-time/`: models trained on 2D Rydberg-atom square lattices of 5x5 atoms, which are prepared with
      adiabatic evolution of 0.4, 0.6, 0.8, 1.0 μs.


### Training models

To train a conditional generative model for the 2D anti-ferromagnetic random Heisenberg model, you can use the
script `heisenberg_train_transformer.py`. An example command is

```shell
python heisenberg_train_transformer.py
```

which uses the default hyperparameters used throughout the paper and trains a model on all the Hamiltonians
sampled in the previous step. Note that this is compute-intensive and should be run with GPU support.
If you want to quickly test on CPUs, you can set the flag `--hamiltonians 1` to use data from only a single Hamiltonian.

### Generating samples from trained models

To generate samples from a trained conditional generative model, you can use the script
`heisenberg_sample_transformer.py`. The flag `--results-dir` indicates the directory pointing to the run package where
the results of a training run have been saved. Note that this should be the root of the results directory and not the
`checkpoints` folder, as this script requires access to flags set during training and saved to the `args.json` file.

### Evaluating properties with classical shadows

Evaluating properties of ground states of the Heisenberg model can be done using the script
`heisenberg_evaluate_properties.py`. Note that this requires you to have completed the previous steps and generated
samples from a trained model using the script `heisenberg_sample_transformer.py`. Similar to the previous step, the flag
`--results-root` indicates the directory pointing to the run package where the results of a training run have been
saved.
The flag `--snapshots` indicates the number of samples (i.e., snapshots) which will be used to estimate the correlation
functions and the entanglement entropies. We use the classical shadow implementation in PennyLane to compute these
properties.


## License

Apache 2.0 

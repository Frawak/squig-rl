# Squig - Reinforcement Learning

This repository resulted from my dive in Reinforcement Learning on the base of 
the Learning-to-Run challenge of [osim-rl](https://github.com/stanfordnmbl/osim-rl).

And I intend to further build on it.

Why "Squig"? Well, the osim running environment showed a armless bipedal walker
and I only know one such creature...

## Setup

The setup is similar to the [osim-rl](https://github.com/stanfordnmbl/osim-rl). 
There just some adjustments:

1) Setup conda: Either download [anaconda](https://www.anaconda.com/download/) or 
install [miniconda](https://conda.io/miniconda.html).

2) Create a conda environment and activate it:
```
conda create -n [Env Name] -c kidzik opensim git python=2.7
source activate [Env Name]
```
Activation under windows:
```
activate [Env Name]
```

3) Get the osim reinforcement learning environment:
```
conda install -c conda-forge lapack git
pip install git+https://github.com/stanfordnmbl/osim-rl.git@v1.5
```
Currently, this repositories runs with the last release which is version 1.5.2
and not the work-in-progress version for the upcoming challenge.

4) Next, keras and one of its backends has to be installed. I am more comfortable
with tensorflow. Make sure to install the CPU version. This repository does not 
contain anything which can be exploited with GPUs and the parallel algorithm 
requires multiple cores. 
```
pip install tensorflow keras
```
So far, it is not a crucial choice which backend you use but in the future, some
backend implementations might be added.

5) Rolling back to previous gym version:
```
pip install gym==0.9.5
```
...because the last osim release uses environment methods from gym before its
clean-up (see [here](https://github.com/stanfordnmbl/osim-rl/issues/92)).

Make sure to install tensorflow for CPUs and not GPUs.

6) Install some additional packages for plotting:
```
conda install matplotlib
conda instal h5py
```

Be aware that I only ran the `trainCollective.py` script on a Linux server. I do
not know if the parallelization is running under Windows as well. The other scripts
are runnable on either OS.



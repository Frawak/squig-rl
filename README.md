# Squig - Reinforcement Learning

This repository resulted from my dive in Reinforcement Learning on the base of 
the Learning-to-Run challenge of [osim-rl](https://github.com/stanfordnmbl/osim-rl).

And I intend to further build on it.

Why "Squig"? Well, the osim running environment showed a armless bipedal walker
and I only know one such creature...

Make sure to read the [acknowledgement section](https://github.com/Frawak/squig-rl/blob/master/README.md#acknowledgement).

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

## Acknowledgement

First off, I have to thank organizers and particpants of the [challenge](https://www.crowdai.org/challenges/nips-2017-learning-to-run).
The exchange throughout it provided an accelerated learning curve.

As a code basis, I chose [keras-rl](https://github.com/keras-rl/keras-rl). At first, 
it was installed as a package and classes were derived from its classes. But 
for my own convenience, the essential code was copied, reduced and modified to suit
my structure (which is not perfect or final, yet). The files, which contain code from 
[keras-rl](https://github.com/keras-rl/keras-rl), begin with `# @keras-rl` and the 
spots are respectively marked with comments. This code falls under the following license:
[https://github.com/Frawak/keras-rl/blob/master/LICENSE](https://github.com/Frawak/keras-rl/blob/master/LICENSE).

A big help (also for many participants of the challenge) was [ctmakro](https://github.com/ctmakro/stanford-osrl). 
Some environment processing was inspired by him. 
He also provides farming code with pyro4, so check it out.

The parallelized approach was refined the one of [fgvbrt](https://github.com/fgvbrt/nips_rl).
I named it 'CollectiveDDPG'...

## A bit more info


The crux in the osim environment are its [time-expensive computations](https://github.com/stanfordnmbl/osim-rl/issues/78).
This drove the participants to refine their reinforcement learning algorithms 
(mainly [DDPG](https://arxiv.org/abs/1509.02971)), e.g. see 
[the paper of fgvbrt and his team](https://arxiv.org/abs/1711.06922).

### Train Collectively

The parallelizations was the most crucial part in my experience. The method used
here, lets `n` workers collect experiences for the main replay buffer and refresh
their policy after each episode with the currently trained actor policy of the 
main process. 

One Tester process documents the progress of the actor policy learning by
executing episodes with the current policy without applying noise.

You will need `2+n, n>=1` cores in order to run `trainCollective.py`.
* 1 for the training process (main process)
* n Explorer (actor/worker)
* 1 Tester 

## Current ToDo list

* Python 3 conversion
* Refine parameter noise (by annealing the probability or introduce a measurement like [fgvbrt](https://github.com/fgvbrt/nips_rl))
* Osim: Add obstacle information to the observation vector
* Some code polishing

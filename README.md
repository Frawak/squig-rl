# Squig - Reinforcement Learning

This repository resulted from my dive in Reinforcement Learning on the base of 
the Learning-to-Run challenge of [osim-rl](https://github.com/stanfordnmbl/osim-rl).

And I intend to further build on and extend this repo.

Why "Squig"? Well, the osim running environment showed a armless bipedal walker
and I only know of one such creature...

Make sure to read the [acknowledgement section](https://github.com/Frawak/squig-rl/blob/master/README.md#acknowledgement).

## Setup

The setup is similar to the [osim-rl](https://github.com/stanfordnmbl/osim-rl). 
There are just some adjustments. 

1) Setup conda: Either download [anaconda](https://www.anaconda.com/download/) or 
install [miniconda](https://conda.io/miniconda.html).

2) Create a conda environment and activate it.

If you want to run osim, use Python 2.7 and the following channels:
```
conda create -n [Env Name] -c kidzik opensim git python=2.7
source activate [Env Name]
```
This will be updated when a new osim version is released. Currently, they are preparing for a new challenge.

If you want to run [gym](https://github.com/openai/gym) or [roboschool](https://github.com/openai/roboschool), 
use Python 3.6. If you want to use Python 3.5, you have to adjust the `Queue` import
in [customDDPG.py](https://github.com/Frawak/squig-rl/blob/master/source/agents/customDDPG.py).
```
conda create -n [Env Name] -c git python=3.6
source activate [Env Name]
```
Activation under windows:
```
activate [Env Name]
```

3) If you want to use Osim, get its reinforcement learning environment:
```
conda install -c conda-forge lapack git
pip install git+https://github.com/stanfordnmbl/osim-rl.git@v1.5
```
Currently, this repositories runs with the last release, which is version 1.5.2,
and not the work-in-progress version for the upcoming challenge.

4) Next, keras and one of its backends has to be installed. I am more comfortable
with tensorflow. Make sure to install the CPU version. This repository does not 
contain anything (yet) which can be exploited with GPUs. Also, the parallel algorithm 
requires multiple cores. Unless you have a bunch of graphics cards lying around...
```
pip install tensorflow keras
```
So far, it is not a crucial choice which backend you use but in the future, some
backend implementations might be added.

5) For Osim: Roll back to a previous gym version:
```
pip install gym==0.9.5
```
...because the last osim release (v1.5) uses environment methods from gym before its
clean-up (see [here](https://github.com/stanfordnmbl/osim-rl/issues/92)).

For Gym or Roboschool, use a more current version.

6) Install some additional packages for plotting:
```
conda install matplotlib
conda install h5py
```

7) If you want to run Roboschool, you have to be more careful with the installation.
Refer: https://github.com/Frawak/squig-rl/blob/master/envs/roboschool.md

Be aware that I only ran the `trainCollective.py` script on a Linux server. I do
not know if the parallelization is running under Windows as well. The other scripts
are runnable on either OS.

## Acknowledgement

First off, I have to thank the organizers and participants of the [challenge](https://www.crowdai.org/challenges/nips-2017-learning-to-run).
The exchange throughout it provided an accelerated learning curve.

As a code basis, I chose [keras-rl](https://github.com/keras-rl/keras-rl). At first, 
it was installed as a package and classes were derived from its classes. But 
for my own convenience, the essential code was copied, reduced and modified to suit
my structure (which is not perfect or final, yet). The files, which contain code from 
[keras-rl](https://github.com/keras-rl/keras-rl), begin with `# @keras-rl` and the 
spots are respectively marked with comments. These code bits fall under the following license:
[https://github.com/Frawak/keras-rl/blob/master/LICENSE](https://github.com/Frawak/keras-rl/blob/master/LICENSE).

A big help (also for many participants of the challenge) was [ctmakro](https://github.com/ctmakro/stanford-osrl). 
Some environment processing was inspired by him. 
He also provides farming code with pyro4, so check it out.

The parallelized approach was refined by the one of [fgvbrt](https://github.com/fgvbrt/nips_rl).
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

But be aware that one process could utilize multiple cores, e.g. the training process through tensorflow.

Watch [this video](https://www.youtube.com/watch?v=9WXPwX7TRZI) for an example
of a training session progress (in the osim running environment). 
Obstacles were disabled. The video shows the policies after each 1000th 
exchange between the Explorers and the main process. The third checkpoint was 
reached after approx. 5 to 6 hours with 10 Explorers. 

## Current ToDo list

* Refine parameter noise (by annealing the probability or introduce a measurement like [fgvbrt](https://github.com/fgvbrt/nips_rl))
* Osim: Add obstacle information to the observation vector
* Some code polishing

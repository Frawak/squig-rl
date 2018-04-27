# Roboschool Setup

1) Create a conda environment and activate it:
```
conda create -n [Env Name] -c kidzik git python=3.5
source activate [Env Name]
```

2) Install gym (maybe an older version, e.g. 0.10.2):
```
pip install gym==0.10.2
```

3) Clone the roboschool repository
```
git clone https://github.com/openai/roboschool.git
cd roboschool/
```

4) If cython is not installed, install it:
```
conda install cython
```

5) To be secure, export path to pkgconfig:
```
export PKG_CONFIG_PATH=/pathToMiniconda/envs/[Env Name]/lib/pkgconfig
```

6) Follow the installation instructions in the [installation section](https://github.com/openai/roboschool#installation). 
If you are working on a server, skip the package installation with `apt`. These packages should be installed. If not, constact a server admin. 

Consider to comment out render after environment step for learning.
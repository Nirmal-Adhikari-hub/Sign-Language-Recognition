# SSM-based Movie genre classification

## Setting Up the Repository
Please run the following commands to set up the repository:
### Create a Conda Environment
```bash
conda create -n video_mamba python=3.10.14
conda activate video_mamba
```
### Installing PyTorch and Other Dependencies
```bash
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```
And we have to install `apex==0.1` through this [URL](https://github.com/NVIDIA/apex), then try code below
```
pip install -r requirements.txt
```
### Installing Mamba Related Packages
```bash
pip install -e causal-conv1d
pip install -e mamba
```
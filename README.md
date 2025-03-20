# How to Run:
!! Still Working It Out !!
Place all illustrations in folder "illustration_dataset"

# Dependencies

$conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

$ conda create -n clip_env python=3.8
$ conda activate clip_env

$ conda install jupyter
$ pip install pillow
$ pip install git+https://github.com/openai/CLIP.git
$ pip install clip-by-openai

## Useful Commands
Check cuda version $ nvcc --version
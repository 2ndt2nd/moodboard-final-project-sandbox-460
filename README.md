# Dependencies
nvcc --version
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install pillow
pip install git+https://github.com/openai/CLIP.git

conda install jupyter
conda create -n clip_env python=3.8
conda activate clip_env
pip install clip-by-openai
## Results

### Vertical to Horizontal
- [Standard Completions](https://www.youtube.com/playlist?list=PLh5XtfDDhGgX1tj4yuPEU-BQUxZnj7KRv)
- [Gao etal.](https://www.youtube.com/playlist?list=PLh5XtfDDhGgVNLC1TfJTBdqYgYshIuyZi)
- [Ours without image shifting](https://www.youtube.com/playlist?list=PLh5XtfDDhGgXDxdyNL4jg8-SpzWCKgEQN)
- [Ours with image shifting](https://www.youtube.com/playlist?list=PLh5XtfDDhGgUsCCemgB8c6lejqQMIdqmf)
- [Ours with both image shifting and post processing](https://www.youtube.com/playlist?list=PLh5XtfDDhGgVonQ0PifxQfr_EG0UpNbul)
    
### Landscape to Ultrawide
- [Standard Completions](https://www.youtube.com/playlist?list=PLh5XtfDDhGgWCs_SpBV6lpcXBoYUmUtrV)
- [Gao etal.](https://www.youtube.com/playlist?list=PLh5XtfDDhGgWF627x6FrpdWzorSZd7yhm)
- [Ours without image shifting](https://www.youtube.com/playlist?list=PLh5XtfDDhGgXgHrKEPTNTpg2dWi8DnB5B)
- [Ours with image shifting](https://www.youtube.com/playlist?list=PLh5XtfDDhGgU8aHSfddl3oXSOwz6yMsB-)
- [Ours with image shifting and post processing](https://www.youtube.com/playlist?list=PLh5XtfDDhGgWBokNQ2bJCV8UK7NeJ7OqB)

## Prerequisites
- Tested on python 3.6.13, ubuntu 18.04 
- Anaconda

## Installation 

```
conda create -n VideoOutpainting python=3.6.13
conda activate VideoOutpainting
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
pip install tensorflow-gpu==1.15
pip install imageio
pip install scikit-image
pip install imageio-ffmpeg
```

- download weights for image completion network from https://github.com/zsyzzsoft/co-mod-gan 
    ./co-mod-gan-places2-050000.pkl

- download weight for RAFT (optical flow) from https://github.com/princeton-vl/RAFT
     ./raft-things.pth

- download weights for COSNet (VOS) from https://github.com/carrierlxk/COSNet
     ./co_attention.pth
	 
## Detailed configuration guide When you use ubuntu 22.04 in 2023
### Prerequisite
- Tested on Ubuntu 22.04, NVidia Tesla v100 
	- Compute capability: 7.0 
	- NVidiai driver 525.125.06, CUDA 12.0
- Mamba-forge-22.9.0-3 with pyenv (pyenv-virtualenv)

#### Install and use gcc-7.5.0, g++-7.5.0
```
sudo add-apt-repository "deb [arch=amd64] http://archive.ubuntu.com/ubuntu focal main universe"
sudo apt update
sudo apt install gcc-7 g++-7
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 7
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 7
```
And select for using gcc-7 and g++-7 with below commands
```
sudo update-alternatives --config gcc
(Select gcc-7)
sudo update-alternatives --config g++
(Select g++-7)
```

And then, Configuration virtual environment with mamba(conda)
```
pyenv install mambaforge-22.9.0-3
pyenv activate mambaforge-22.9.0-3
mamba create -n VideoOutpainting python=3.6.13
mamba activate VideoOutpainting
mamba install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 cudatoolkit-dev=10.1 cudnn=7.6 matplotlib tensorboard scipy opencv -c pytorch
pip install tensorflow-gpu==1.15
pip install imageio
pip install scikit-image
pip install imageio-ffmpegn
```
If device_lib.list_local_devices() doesn't work properly on 'tool/dnnlib/tflib/custom_ops.py', 
(Possible to check like below)
```
python
>>> from tensorflow.python.client import device_lib
>>> device_lib.list_local_devices()
```
(If you see the message like 'ImportError: libcu*.so.10.0: cannot open shared object file: No such file or directory')
you have to work around like below,
```
(Move to your virtual environment folder)
cd ~/.pyenv/versions/mambaforge22.9.0-3/VideoOutpainting/lib64
(Make symbolic link for missing libcu*.so.10.0 libraries)
ln -s libcusparse.so libcusparse.so.10.0
ln -s libcusolver.so libcusolver.so.10.0
ln -s libcurand.so libcurand.so.10.0
ln -s libcufft.so libcufft.so.10.0
ln -s libcublas.so libcublas.so.10.0
ln -s libcudart.so libcudart.so.10.0
```
And then, this code will work in your conda(mamba) environment 'VideoOutpainting'.

### References for this trouble shooting
- https://developer.nvidia.com/cuda-gpus
- https://linuxconfig.org/how-to-switch-between-multiple-gcc-and-g-compiler-versions-on-ubuntu-22-04-lts-jammy-jellyfish
- https://bo-10000.tistory.com/75
- https://askubuntu.com/questions/1406962/install-gcc7-on-ubuntu-22-04
- https://bigdata-analyst.tistory.com/328


## Usage


- Video outpainting on a single video:
```bash
cd tool
python video_outpaint.py --path ../frames/ --outroot ../results/frames/ --Width 0.125 --replace
```
replace: remove and recomplete 0.125*the width of the video on each side.

no replace: extrapolate 0.125*the width of the video width on each side

- Run dataset:
```bash
cd tool
python runDataset.py --pathToDataset /home/user/Documents/DAVIS-data/DAVIS/JPEGImages/480p/ --outroot ../result/ --vertical
```
vertical: Vertical to horizontal video conversion (0.33)

no vertical: horizontal to ultra-wide video conversion (0.125)
## Acknowledgments
- Our code is based upon [FGVC](https://github.com/vt-vl-lab/FGVC/).
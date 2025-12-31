# Dump of commands I used to set up my Lambda GPU. I executed these manually via SSH.
# I probably installed too much, so I should revisit this eventually.

# Install nvidia-smi
VERSION=570
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-utils-${VERSION}-server
sudo apt install nvidia-driver-${VERSION}-server

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version

# python
uv install python 3.13
uv init .
uv add torch pydantic numpy

# CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-1

# reboot
sudo reboot

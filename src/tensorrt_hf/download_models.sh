curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-1.0-tensorrt
cd stable-diffusion-xl-1.0-tensorrt
git lfs pull
cd ..


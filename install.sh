conda create -n easyr1-wxr python=3.11
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install ninja
pip install -r requirements.txt
conda env update -f environment.yml
pip install flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

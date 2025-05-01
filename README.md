# SayCan
This is a forked repo to organize the official [ipynb](https://github.com/google-research/google-research/tree/master/saycan) implementation of SayCan ([Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691)) for easier further research. 

This fork enables a cli-like layer and also enables support
for open-source llms like llama-3.1 supported via the transformers library.

## Testing
This repository has been tested to be working with Ubuntu 22.04.2 and
pip3==20.2.3 on the conda environment (instructions below).


## 1. Setup Environment

Clone this repo. Create and activate new conda environment with python 3.9 as
follows.
```
conda create -n saycan python=3.9.1
conda activate saycan
```

### Downgrade pip and install all required packages
```
pip3 install pip==20.2.3
pip3 install -r requirements.txt
```

### For using open-source models
Ensure that you setup vllm and register your huggingface token.

## 2. Download relevant data
The instructions below try downloading from official sources. If there are
any problems there, I also host the assets/ directory via this shared [link](https://arizonastateu-my.sharepoint.com/:u:/g/personal/rkaria_sundevils_asu_edu/Ea0EWvHawsxJgAQ5Zp-dVk8BqRGcds9B7onbQ--wu1OEVg?e=Hm9dZn).
Simply download it and unzip it in the project root directory: `saycan.ROOT_DIR`

If you still have issues (eg. broken links), email me by finding my email on
my personal webpage [rushangkaria.github.io](https://rushangkaria.github.io)

### 2.1 Download PyBullet assets.
```
mkdir assets/
gdown -O assets/ 1Cc_fDSBL6QiDvNT4dpfAEbhbALSVoWcc
gdown -O assets/ 1yOMEm-Zp_DL3nItG9RozPeJAmeOldekX
gdown -O assets/ 1GsqNLhEl9dd4Mc3BM0dX3MibOI1FVWNM
unzip assets/ur5e.zip -d assets/
unzip assets/robotiq_2f_85.zip -d assets/
unzip assets/bowl.zip -d assets/
```
### 2.2 Download ViLD pretrained model weights.
```
gsutil cp -r gs://cloud-tpu-checkpoints/detection/projects/vild/colab/image_path_v2 assets/
```

### 2.3 Download pregenerated dataset.
You can skip this process if you want to generate data by yourself with `gen_data.py`.
Download pregenerated dataset by running

```
gdown -O assets/ 1yCz6C-6eLWb4SFYKdkM-wz5tlMjbG2h8
```
### 2.4 Download pretrained low-level policy.
```
gdown -O assets/ 1Nq0q1KbqHOA5O7aRSu4u7-u27EMMXqgP
```

### 3. You are all set!
Don't forget to add your openai key in `llm.py`.
If you have downloaded the pretrained policy in 2.4, you can now run `demo.py` to visualize the evaluation process.
If you want to train a model from scratch, run `train.py`.

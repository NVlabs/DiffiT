# DiffiT: Diffusion Vision Transformers for Image Generation

Official PyTorch implementation of [**DiffiT: Diffusion Vision Transformers for Image Generation**](https://arxiv.org/abs/2312.02139).


For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

[![Star on GitHub](https://img.shields.io/github/stars/NVlabs/DiffiT.svg?style=social)](https://github.com/NVlabs/DiffiT/stargazers)

**DiffiT** (Diffusion Vision Transformers) is a generative model that combines the expressive power of diffusion models with Vision Transformers (ViTs), introducing **Time-dependent Multihead Self Attention (TMSA)** for fine-grained control over the denoising at each timestep. DiffiT achieves SOTA performance on class-conditional ImageNet generation at multiple resolutions, notably an **FID score of 1.73** on ImageNet-256.

![teaser](./assets/imagenet.png)


![teaser](./assets/latent_diffit.png)

## 💥 News 💥
- **[03.08.2026]** 🔥🔥 DiffiT code and pretrained model are released !
- **[07.01.2024]** 🔥🔥 DiffiT has been accepted to [ECCV 2024](https://eccv.ecva.net/) !
- **[04.02.2024]**  Updated [manuscript](https://arxiv.org/abs/2312.02139) now available on arXiv !
- **[12.04.2023]** 🔥 Paper is published on arXiv !


## Models 


### ImageNet-256

| Model| Dataset |  Resolution | FID-50K | Inception Score | Download |
|---------|----------|-----------|---------|--------|--------|
|**DiffiT** | ImageNet | 256x256   | **1.73**    | **276.49**|[model](https://huggingface.co/nvidia/DiffiT/resolve/main/diffit_256.safetensors)|

### ImageNet-512

| Model| Dataset |  Resolution | FID-50K | Inception Score | Download |
|---------|----------|-----------|---------|--------|--------|
|**DiffiT** | ImageNet | 512x512   | **2.67**    | **252.12**|[model](https://huggingface.co/nvidia/DiffiT/resolve/main/diffit_512.safetensors)|


## Getting Started: Sampling & Evaluation

This repository provides the code for the DiffiT model, pretrained model checkpoints, and everything needed to sample images and compute FID scores to reproduce the results reported in our paper.

### Sampling Images

Image sampling is performed using `sample.py`. To reproduce the reported numbers, use the commands below.

**ImageNet-256:**

```bash
python sample.py \
    --log_dir $LOG_DIR \
    --cfg_scale 4.4 \
    --model_path $MODEL \
    --image_size 256 \
    --model Diffit \
    --num_sampling_steps 250 \
    --num_samples 50000 \
    --cfg_cond True
```

**ImageNet-512:**

```bash
python sample.py \
    --log_dir $LOG_DIR \
    --cfg_scale 1.49 \
    --model_path $MODEL \
    --image_size 512 \
    --model Diffit \
    --num_sampling_steps 250 \
    --num_samples 50000 \
    --cfg_cond True
```

We also provide ready-to-use Slurm scripts for convenience:
- `slurm_sample_256.sh` — samples 50K images at 256×256 resolution
- `slurm_sample_512.sh` — samples 50K images at 512×512 resolution

### Computing FID

Once images have been sampled, you can compute the FID and other metrics using the provided `eval_run.sh` script. Our evaluation pipeline exactly follows the protocol from [openai/guided-diffusion/evaluations](https://github.com/openai/guided-diffusion/tree/main/evaluations).

```bash
bash eval_run.sh
```

### Expected Results

Running the above sampling and evaluation commands should yield the following metrics:

**ImageNet-256:**

| Inception Score | FID | sFID | Precision | Recall |
|:-:|:-:|:-:|:-:|:-:|
| 276.49 | 1.73 | 4.54 | 0.8024 | 0.6205 |

**ImageNet-512:**

| Inception Score | FID | sFID | Precision | Recall |
|:-:|:-:|:-:|:-:|:-:|
| 252.13 | 2.67 | 4.99 | 0.8277 | 0.5500 |

> **Note:** Small variations in the reported numbers are expected depending on the device used for sampling and due to numerical precision differences.


## Citation

```
@inproceedings{hatamizadeh2025diffit,
  title={Diffit: Diffusion vision transformers for image generation},
  author={Hatamizadeh, Ali and Song, Jiaming and Liu, Guilin and Kautz, Jan and Vahdat, Arash},
  booktitle={European Conference on Computer Vision},
  pages={37--55},
  year={2025},
  organization={Springer}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=NVlabs/DiffiT&type=Date)](https://star-history.com/#NVlabs/DiffiT&Date)

## Licenses

Copyright © 2026, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Click [here](LICENSE) to view a copy of this license.

The pre-trained models are shared under [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.


## Acknowledgement
We gratefully acknowledge the authors of [Guided-Diffusion](https://github.com/openai/guided-diffusion/tree/main/), [DiT](https://github.com/facebookresearch/DiT/tree/main) and [MDT](https://github.com/sail-sg/MDT/tree/mdtv1) for making their excellent codebases publicly available.

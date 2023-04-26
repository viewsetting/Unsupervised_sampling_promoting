# Unsupervised Sampling Promoting for Stochastic Human Trajectory Prediction

[![arXiv](https://img.shields.io/badge/arXiv-2304.04298-b31b1b.svg?style=flat)  ](https://arxiv.org/abs/2304.04298)[![arXiv](https://img.shields.io/badge/CVPR-2023-1c75b8)]()



The official repo for "Unsupervised Sampling Promoting for Stochastic Human Trajectory Prediction" (accepted by **CVPR 2023**)

**[Guangyi Chen](https://chengy12.github.io/)\*, [Zhenhao Chen](https://zhenhaochenofficial.github.io/)\*, Shunxing Fan, [Kun Zhang](https://www.andrew.cmu.edu/user/kunz1/)** \*\* 

> **Abstract**
>
> The indeterminate nature of human motion requires trajectory prediction systems to use a probabilistic model to formulate the multi-modality phenomenon and infer a finite set of future trajectories. However, the inference processes of most existing methods rely on Monte Carlo random sampling, which is insufficient to cover the realistic paths with finite samples, due to the long tail effect of the predicted distribution. To promote the sampling process of stochastic prediction, we propose a novel method, called BOsampler , to adaptively mine potential paths with Bayesian optimization in an unsupervised manner, as a sequential design strategy in which new prediction is dependent on the previously drawn samples. Specifically, we model the trajectory sampling as a Gaussian process and construct an acquisition function to measure the potential sampling value. This acquisition function applies the original distribution as prior and encourages exploring paths in the long-tail region. This sampling method can be integrated with existing stochastic predictive models without retraining. Experimental results on various baseline methods demonstrate the effectiveness of our method.

\* *Authors contributed equally and are listed alphabetically by first name*

** *Code & Configuration on the exception subset will be updated later.*

## Introduction

We prepare the implementation of BOsampler on PECnet baseline as an example. It is easy to extend our code to other baseline models, too. The default settings has been verified to run on NVIDIA A6000 (48G VRAM) and NVIDIA RTX 3090 (24G VRAM). To run on lower-end GPUs, you might have to reduce the number of ``batch_size`` in ``./configs/pecnet/bo.yaml``

## Requirements

```
PyTorch==1.11.0
BoTorch==0.6.4
numpy,easydict,pyyaml
```

### Prepare the running environment using conda (Recommended)

We recommend using conda to create a virtual environment to run BOsampler.

First, create a new conda environment.

```
conda create -n bosampler python=3.8 
```

Activate the conda environment just created.

```
conda activate bosampler
```

Install PyTorch using official channel.

```
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch
```

Use pip to install other requirements.

```
pip install easydict pyyaml numpy botorch==0.6.4
```

## How to test

We prepare three shell scripts for testing. ``run_mc.sh`` for Monte Carlo method, ``run_qmc.sh`` for Quasi-Monte Carlo method, and ``run_bo.sh`` for BOsampler method. You can use a unix shell like ``/bin/sh`` to exexute any of those scripts as you like.

### Run BOsampler

```
sh run_bo.sh
```

## Adjust for other hyperparameter configurations

We store the configuration files under ``./configs``. You can explore other settings as you like by editing those configuration files.

## BibTex

If you find our work helpful, please cite us by adding the following BibTex.

```latex
@article{chen2023unsupervised,
  title={Unsupervised Sampling Promoting for Stochastic Human Trajectory Prediction},
  author={Chen, Guangyi and Chen, Zhenhao and Fan, Shunxing and Zhang, Kun},
  journal={arXiv preprint arXiv:2304.04298},
  year={2023}
}
```

## Acknowledgement

This project was partially supported by the **National Institutes of Health (NIH)** under Contract R01HL159805, by the **NSF-Convergence Accelerator** Track-D award #2134901, by a grant from **Apple Inc.**, a grant from **KDDI Research Inc.**, and generous gifts from **Salesforce Inc.**, **Microsoft Research**, and **Amazon Research**. 

We would like to thank our colleague **Zunhao Zhang** from *MBZUAI* for providing computation resource for part of the experiments.


# Differential Privacy

Implements differential privacy based optimization algorithms for deep learning.

This includes a differentially private SGD optimizer (DPSGD) based on Abadi et al. and perturbed iterative gradident descent optimizers (PIGDO) based on Ding et al. which includes PIAdam, PIAdaGrad and PIRMSProp.

## Introduction

Deep Learning is widely being used to solve various problems in different domains with exceptional results. However while applying it to various real life situations, privacy of individuals is a major concern. Training data usually contains sensitive data about people and training models on such a dataset can leak their personal information. Thus, it is important to incorporate privacy preserving measures which protect PII information from getting leaked.

Differential privacy is the most widely used privacy preserving concept introduced by Dwork et al. It is a rigorous framework which gives a strong guarantee of protecting an individual's privacy, making it difficult for an adversary to determine its presence or absence in the dataset. The main idea is to add some sort of noise in the training process which masks an individual's presence.

## Method

Abadi et al. introduced Differentially Private Stochastic Gradient Descent (DP - SGD) to add differential privacy in the optimization part of the training itself. In this way, training any model using this optimizer will enforce $(\epsilon , \delta)$-differential privacy, where $\epsilon$ and $\delta$ are the privacy budget and privacy parameter. The main idea is to clip the gradients of the model (per lot) and then add gaussian noise to the clipped gradients. This will ensure that in every lot of data, gradients will not contain exact information about the individuals in that lot. The scale of gaussian noise added and the maximum norm of gradients permitted determine the privacy budget.

Ding et al. introduces perturbed gradients in optimizers like Adagrad, RMSprop and Adam. Gaussian noise is added to the clipped gradients in each iteration of the optimizer, thus ensuring $(\epsilon , \delta)$-differential privacy.

Also implements the privacy accountants of both the papers which keep track of the privacy budget ($\epsilon$) spent over each iteration of the training process.

## Setup

use `pip install -r requirements.txt` to install the required packages. Python 3.11 is required.

## References

Abadi, Martín, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. “Deep Learning with Differential Privacy.” In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security, 308–18, 2016. https://doi.org/10.1145/2976749.2978318.  
  
Ding, Xiaofeng, Lin Chen, Pan Zhou, Wenbin Jiang, and Hai Jin. “Differentially Private Deep Learning with Iterative Gradient Descent Optimization.” ACM/IMS Transactions on Data Science 2, no. 4 (November 30, 2021): 1–27. https://doi.org/10.1145/3491254.  
  
“Per-Sample-Gradients — PyTorch Tutorials 2.2.1+Cu121 Documentation,” Pytorch.org, 2024, https://pytorch.org/tutorials/intermediate/per_sample_grads.html.

Cynthia Dwork and Aaron Roth, “The Algorithmic Foundations of Differential Privacy,” Foundations and Trends in Theoretical Computer Science 9, no. 3-4 (January 1, 2013): 211–407, https://doi.org/10.1561/0400000042.

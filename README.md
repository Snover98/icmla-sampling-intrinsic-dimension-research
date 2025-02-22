# icmla-sampling-intrinsic-dimension-research
Reichman University Data Streaming Algorithms and Online Learning Final Project

## Summary
This project aims to expand on the paper
"Finding an Optimal Small Sample of Training Dataset for Computer Vision Deep-Learning Model"[[1]](#1)
by investigating the relationship between the dimension the dataset samples are reduced to via JLS
and the intrinsic dimension of the dataset as described by
"The intrinsic dimension of images and its impact on learning"[[2]](#2).

## What will be researched
Since we are expanding on [[1]](#1), we'll evaluate performance on MNIST[[3]](#3), CIFAR-10[[4]](#4), and SVHN[[5]](#5).
In addition, since [[1]](#1) saw most of the improvement on a dataset size of $s=100$, we'll use that size for all of our experiments.
(TODO: revisit later)

### Choice of Intrinsic Dimension per Dataset
In Table 1 in section 6 of [[2]](#2) we can see the following dimension estimates given $k$:

| Dataset  | MNIST | SVHN | CIFAR-10 |
|----------|-------|------|----------|
| $k = 3$  | 7     | 9    | 13       |
| $k = 5$  | 11    | 14   | 21       |
| $k = 10$ | 13    | 18   | 25       |
| $k = 20$ | 19    | 19   | 26       |

From what I've seen in the paper, $k=5$ seems to have the best trade-off between variance and bias,
so we'll start from the values matching it (TODO: revisit later)

| Dataset  | Intrinsic Dimension |
|----------|---------------------|
| MNIST    | 11                  |
| SVHN     | 14                  |
| CIFAR-10 | 21                  |

### Choices of $d_{JL}$ given $d_{ID}$
Given an intrinsic dimension of $d_{ID}$,
we'll want to test results for the following choices of $d_{JL}$:
* $d_{JL} = d_{ID}$
* $d_{JL} = 2d_{ID}$
* $d_{JL} = 5d_{ID}$
* $d_{JL} = d_{ID}ln\left(d_{ID}\right)$
* $d_{JL} = d_{ID}log2\left(d_{ID}\right)$
* $d_{JL} = d_{ID}^{1.5}$
* $d_{JL} = d_{ID}^{1.5}ln\left(d_{ID}\right)$
* $d_{JL} = d_{ID}^2$

(all values will be rounded when evaluating them)

## References
<a id="1">[1]</a> Yehezkel, A.,  Elyashiv E. (2024).
Finding an Optimal Small Sample of Training Dataset for Computer Vision Deep-Learning Models

<a id="2">[2]</a> Pope P, Zhu C, Abdelkader A, Goldblum M, Goldstein T: The Intrinsic Dimension of 
Images and Its Impact on Learning. arXiv \[csCV\] 2021.

<a id="2">[3]</a>  L. Deng, ”The mnist database of handwritten digit images for machine
 learning research,” IEEE Signal Processing Magazine, 2012.

<a id="2">[4]</a> A. Krizhevsky, ”Learning Multiple Layers of Features from Tiny Im
ages,” 2009.

<a id="2">[5]</a> Y. Netzer, T. Wang, A. Coates, A. B. Wu and A. Y. Ng, ”Reading
 Digits in Natural Images with Unsupervised Feature Learning,” NIPS
 Workshop on Deep Learning and Unsupervised Feature Learning, 2011.
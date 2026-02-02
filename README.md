# CASGNN

# Abstract
> Multivariate time series forecasting often requires capturing complex cross-series dependencies while remaining computationally efficient for large-scale and long-horizon applications. Existing pure graph-based models effectively characterize local relational patterns but struggle with long-range structures, whereas transformer-based architectures incur substantial computational overhead. To overcome these limitations, we propose CASGNN, a lightweight community-aware spectral graph model built upon three key components: (i) Fourier-domain representations that expose global temporal structures in the frequency space, (ii) community-aware message passing that enables relational reasoning among locally coherent groups of time series, and (iii) a low-pass filtering mechanism that suppresses high-frequency noise and enhances stability for long-horizon forecasting. Extensive experiments on 16 real-world datasets demonstrate that CASGNN consistently achieves state-of-the-art or highly competitive accuracy, outperforming strong transformer-, linear-, and graph-based baselines in both short- and long-term forecasting. Moreover, CASGNN delivers these improvements with significantly lower computational cost, exhibiting the smallest parameter footprint and most stable training time across varying sequence lengths and data sizes. Together, these results show that CASGNN strikes an effective balance between prediction accuracy and computational efficiency, making it a practical and scalable solution for modern time-series forecasting workloads.

# Dependencies
- Python 3.9.23
- torch_Geometric 2.6.1
- torch 2.2.1
- dgl 2.4.0+cu118
- networkx 3.1
- igraph 0.11.9
- numpy 1.24.4

# Dataset
Dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/15JmxFLPWgvrq4QGanpdmIlAZGdmJ39At/view?usp=sharing).

# Acknowledgment
This code is implemented based on FourierGNN (https://github.com/aikunyi/FourierGNN.git) and UFGTime (https://github.com/WonderHeiYi/UFGTIME). Please refer to both for implementation details.

# Run
```
python main.py --dataset [dataset]
```

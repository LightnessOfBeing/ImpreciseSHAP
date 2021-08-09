# Imprecise SHAP

[![arXiv](https://img.shields.io/badge/arXiv-2106.09111-<COLOR>.svg)](https://arxiv.org/abs/2106.09111)
[![paperswithcode](https://img.shields.io/badge/paperswithcode-ImpreciseSHAP-teal)](https://paperswithcode.com/paper/an-imprecise-shap-as-a-tool-for-explaining)

Implementation of the algorithm described in the paper [An Imprecise SHAP as a Tool for Explaining the Class Probability Distributions under Limited Training Data](https://arxiv.org/abs/2106.09111)

You can find an example of how to run code in the **example.ipynb**.

All necessary packages are located in **requirements.txt**.

# Abstract

One of the most popular methods of the machine learning prediction explanation is the SHapley Additive exPlanations method (SHAP). An imprecise SHAP as a modification of the original SHAP is proposed for cases when the class probability distributions are imprecise and represented by sets of distributions. The first idea behind the imprecise SHAP is a new approach for computing the marginal contribution of a feature, which fulfils the important efficiency property of Shapley values. The second idea is an attempt to consider a general approach to calculating and reducing interval-valued Shapley values, which is similar to the idea of reachable probability intervals in the imprecise probability theory. A simple special implementation of the general approach in the form of linear optimization problems is proposed, which is based on using the Kolmogorov-Smirnov distance and imprecise contamination models. Numerical examples with synthetic and real data illustrate the imprecise SHAP.

# Citation

Please use this bibtex if you want to cite this work in your publications:

```
@article{DBLP:journals/corr/abs-2106-09111,
  author    = {Lev V. Utkin and
               Andrei V. Konstantinov and
               Kirill A. Vishniakov},
  title     = {An Imprecise {SHAP} as a Tool for Explaining the Class Probability
               Distributions under Limited Training Data},
  journal   = {CoRR},
  volume    = {abs/2106.09111},
  year      = {2021},
  url       = {https://arxiv.org/abs/2106.09111},
  archivePrefix = {arXiv},
  eprint    = {2106.09111},
  timestamp = {Tue, 29 Jun 2021 16:55:04 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2106-09111.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

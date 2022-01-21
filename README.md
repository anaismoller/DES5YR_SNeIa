<!-- [![arXiv](https://img.shields.io/badge/arxiv-astro--ph%2F1901.06384-red)](https://arxiv.org/abs/XXX)  -->


### DES 5-year photometrically identified SNe Ia using host galaxy redshifts

Code to reproduce analysis in "DES 5-year photometrically identified SNe Ia using host galaxy redshifts" Möller et al. 2022.

We use the photometric classification framework [SuperNNova](https://github.com/supernnova/SuperNNova) [(Möller & de Boissière 2019)](https://academic.oup.com/mnras/article-abstract/491/3/4277/5651173) trained on realistic DES-like simulations to classify DES 5-year data and obtain SNe Ia for astrophysics and cosmology analysis.

This repository contains:
- ./reproduce/Pippin* : configuration files to recreate simulations used in the analysis using [pippin](https://github.com/dessn/Pippin) (Hinton & Brout 2020) analysis pipeline
- ./reproduce/ : shell files to launch SuperNNova database, training and testing with simulations and data
- *.py: analysis reproduction codes (prints outputs quoted in paper, saves plots and samples)


[![report](https://img.shields.io/badge/arxiv-report-green)](https://arxiv.org/pdf/2003.14286.pdf)

# GeomFmaps-pytorch
A minimalist pytorch implementation of: "Deep Geometric Functional Maps: Robust Feature Learning for Shape Correspondence" [[1]](#bookmark-references), appeared in [CVPR 2020](http://cvpr2020.thecvf.com/).

## Installation
This implementation runs on python >= 3.7, use pip to install dependencies:
```
pip3 install -r requirements.txt
```

## Download data & preprocessing
The preprocessing code will be added later.
For the moment, we refer the reader to the [original implementation](https://github.com/LIX-shape-analysis/GeomFmaps) of GeomFmaps to download the data and the preprocessing code.

It should be noted that for each dataset (faust, scape, etc), this module expect that the dataset folder contains 3 folders:

 * `off` folder: this folder contains the meshes
 * `spectral` folder: this folder contains the laplace beltrami related data. It's composed from files having the same name as the `off` folder. Each fileis a `.mat` contaning  a `dict` containing three keys: `evals`, `evecs` and `evecs_trans`. This files are created by the preprocessing code.
 * `corres` folder: this folder contains the ".vts" files necessary for the calculation of the ground truth maps.

## Usage
Use the `config.yaml` file to specify the hyperparameters as well as the dataset to be used.

Use the `train.py` script to train the GeomFmaps model.
```
python3 train.py
```

References
---------------------
[1] [Deep Geometric Functional Maps: Robust Feature Learning for Shape Correspondence](https://arxiv.org/pdf/2003.14286.pdf)
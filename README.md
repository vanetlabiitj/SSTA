# Spatio-temporal Sparse Attack on Adaptive Graph Neural Networks for Traffic Forecasting

## Environment 
* [PyTorch](https://pytorch.org/) (tested on 1.8.0)
* [mmcv](https://github.com/open-mmlab/mmcv)

## Datasets
We use the METR-LA and PeMS-Bay datasets ([link](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX)). 

## Usage
To train and evaluate a baseline model, run the following commands:
```
# Grey-box attack  METR-LA
python grey_rain.py configs/SemiMETRLA/SemiMETRLA-train0.7-val0.1-test0.2-in12out12-0.1nodes-stawnet-standard.yaml
python grey_test.py configs/SemiMETRLA/SemiMETRLA-train0.7-val0.1-test0.2-in12out12-0.1nodes-stawnet-standard.yaml -a grey-attack

# Grey-box attack PeMS-Bay
python grey_train.py configs/SemiPeMS/PeMS-train0.7-val0.1-test0.2-in12out12-0.1nodes-gwnet-standard.yaml
python grey_test.py configs/SemiPeMS/PeMS-train0.7-val0.1-test0.2-in12out12-0.1nodes-gwnet-standard.yaml -a grey-attack

# Black-box attack  METR-LA
python grey_test.py configs/BlackMETRLA/SemiMETRLA-train0.7-val0.1-test0.2-in12out12-0.1nodes-stawnet-standard.yaml -a black-attack

# Black-box attack PeMS-Bay
python grey_test.py configs/BlackPeMS/BlackMETRLA-train0.7-val0.1-test0.2-numsteps12-0.1nodes-source_stawnet-target_astgcn.yaml -a black-attack

```
Here `-a ALL` ,`-a grey-attack`, and `-a black-attack` denote that we evaluate attacks including STPGD-TNDS, STMIM-TNDS, PGD-Random, PGD-PR, PGD-Centrality, PGD-Degree, MIM-Random, MIM-PR, MIM-Centrality, MIM-Degree .

## Environment
```
conda env create -f pytorch.yaml

## Acknowledgement
We thank the authors for the following repositories for code reference:
[Practical Adversarial Attacks on Spatiotemporal Traffic Forecasting Models], [Robust Spatiotemporal Traffic Forecasting with Reinforced Dynamic Adversarial Training] etc.


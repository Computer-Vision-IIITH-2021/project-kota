# Project 7
- Topic - Image Super Resolution
- Paper - Learning a Single Convolutional Super-Resolution Network for
Multiple Degradations
- Paper Link - https://arxiv.org/pdf/1712.06116v2.pdf


# Run Instructions
1. `make run`

# Usage
## Requirements
- `python=3.8`
- `pytorch=1.7.1`
- `cudatoolkit=10.2`

## Code structure
```
.
├── docs
    |__ TeamKota-MidEvalReport.pdf
    |__ Project Proposal
│       ├── Method.pdf
│       └── TeamKota-ProjectProposal.pdf
├── README.md
├── Makefile
├── data
├── environment.yml
└── src
    ├── dataset.py
    ├── kernels.py
    ├── logger.py
    ├── main.py
    ├── model.py
    ├── train.py
    └── utils.py
```
## Setup env
- `conda env create --file environment.yaml`

## Data

- use `Name` in main.py

| Id | Name |Info      | Link |
| --- | --- | ----------- | ----------- |
| 1 | DIV2K_train_HR | HR images  | http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip       |
| 2 | DIV2K_train_LR_bicubic_X2 | Bicubic x2 downscaling  | http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip       |

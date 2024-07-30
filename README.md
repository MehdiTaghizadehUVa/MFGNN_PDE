# Multifidelity Graph Neural Networks for Efficient and Accurate Mesh-Based PDE Surrogate Modeling

## Overview

This repository contains the implementation of the multifidelity graph neural networks (MFGNN) for efficient and accurate mesh-based partial differential equations (PDE) surrogate modeling. The methods presented here aim to overcome the computational challenges associated with traditional numerical methods such as finite element methods (FEM) and finite volume methods (FVM) by integrating multifidelity strategies with graph neural networks (GNNs).

Our approach leverages both low-fidelity and high-fidelity data to train more accurate surrogate models for mesh-based PDE simulations, significantly reducing the computational demands while improving robustness and generalizability.

## Features

- **Hierarchical Multifidelity GNN (MFGNN_H)**: Combines coarse-to-fine mesh strategies with weighted k-nearest neighbors for upsampling.
- **Curriculum Learning-Based MFGNN (MFGNN_CL)**: Gradually introduces training data from low-fidelity to high-fidelity, enhancing learning efficiency and model performance.
- **Standard Fidelity GNN (SFGNN)**: Provides a baseline model for comparison.

## Installation

To install the necessary dependencies, run:
```bash
pip install torch dgl numpy matplotlib scikit-learn wandb

## Usage

### Data Preparation

Prepare your training and test datasets as required by the GNN models. The data should be in a format where nodes represent mesh points and edges represent the connectivity between these points. Ensure the dataset directories are correctly specified in the `Constants` classes within each script.

### Training

To train the MFGNN models, use the following scripts:

- **MFGNN_H Training**:
  ```bash
  python MFGNN_H.py

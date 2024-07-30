# Multifidelity Graph Neural Networks for Efficient and Accurate Mesh-Based PDE Surrogate Modeling

## Overview

This repository contains the implementation of the multifidelity graph neural networks (MFGNN) for efficient and accurate mesh-based partial differential equations (PDE) surrogate modeling. The methods presented here aim to overcome the computational challenges associated with traditional numerical methods such as finite element methods (FEM) and finite volume methods (FVM) by integrating multifidelity strategies with graph neural networks (GNNs).

Our approach leverages both low-fidelity and high-fidelity data to train more accurate surrogate models for mesh-based PDE simulations, significantly reducing the computational demands while improving robustness and generalizability.

## Features

- **Hierarchical Multifidelity GNN (MFGNN_H)**: Combines coarse-to-fine mesh strategies with weighted k-nearest neighbors for upsampling.
- **Curriculum Learning-Based MFGNN (MFGNN_CL)**: Gradually introduces training data from low-fidelity to high-fidelity, enhancing learning efficiency and model performance.
- **Single Fidelity GNN (SFGNN)**: Provides a baseline model for comparison.

## Installation

To install the necessary dependencies, run:

`pip install torch dgl numpy matplotlib scikit-learn wandb modulus`

## Usage

### Data Preparation

Prepare your training and test datasets as required by the GNN models. The data should be in a format where nodes represent mesh points and edges represent the connectivity between these points. Ensure the dataset directories are correctly specified in the `Constants` classes within each script.

### Data Generation

Two scripts are provided in the `Data_Generation` folder for generating training data using ANSYS MAPDL:

- **Notched Plate Data Generation**: This script generates training data for a notched rectangular steel plate with specified force and ratio ranges. It uses both low-fidelity and high-fidelity mesh refinements.

- **Variable Hole Plate Data Generation**: This script generates training data for a steel plate with a variable hole position and size, ensuring the hole is completely within the plate. It also uses both low-fidelity and high-fidelity mesh refinements.

### Training

To train the MFGNN models, use the following scripts:

- **MFGNN_H Training**:
  
`python MFGNN_H.py`

- **MFGNN_CL Training**:
  
`python MFGNN_CL.py`

- **SFGNN Training**:
  
`python SFGNN.py`

### Evaluation

After training, evaluate the models using the `post_training_analysis_and_plotting` function within each script. You can directly run the scripts to perform evaluation and generate plots:

- **Evaluate MFGNN_H**:

  `python MFGNN_H.py`

- **Evaluate MFGNN_CL**:

  `python MFGNN_CL.py`

- **Evaluate SFGNN**:
  
  `python SFGNN.py`

## Experiments

### Stress Distribution in 2D Plates

The proposed methodologies were validated by assessing stress concentration in notched rectangular steel plates and plates with a hole. The models consistently demonstrated superior performance compared to single-fidelity GNN models, significantly reducing computational costs while achieving higher accuracy.

### Vehicle Aerodynamics Simulation

The models were further validated using industry-level vehicle aerodynamics simulations with Ahmed body geometries. The MFGNN models outperformed traditional methods, showing substantial improvements in accuracy and parameter efficiency.

## Results

The results from various experiments indicate that the proposed MFGNN models can achieve significant reductions in computational costs and improved accuracy over traditional single-fidelity models. Detailed performance metrics and comparisons are available in the publication associated with this repository.

## Publication

For a comprehensive understanding of the methodologies and results, please refer to our publication:

Taghizadeh, M., Nabian, M. A., & Alemazkoor, N. (2024). Multifidelity Graph Neural Networks for Efficient and Accurate Mesh-Based Partial Differential Equations Surrogate Modeling. *Computer-Aided Civil and Infrastructure Engineering*. [DOI: 10.1111/mice.13312](https://doi.org/10.1111/mice.13312)

## Contact

For any questions or issues, please contact Mehdi Taghizadeh at jrj6wm@virginia.edu.

## License

This project is licensed under the terms of the Creative Commons Attribution-NonCommercial-NoDerivs License. See [LICENSE](LICENSE) for more details.

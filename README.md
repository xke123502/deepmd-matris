# DeepMD-MatRIS

A plugin that integrates [MatRIS](https://github.com/your-repo/MatRIS) with [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit) for machine learning interatomic potentials.

## Overview

This plugin allows you to use MatRIS models within the DeePMD-kit ecosystem, leveraging DeePMD-kit's training infrastructure, data handling, and deployment tools.
This repository is also refered to [Deepmd-GNN].

## Installation

### Prerequisites

1. Install DeePMD-kit with PyTorch backend:
```bash
pip install deepmd-kit[torch]
```

2. Install MatRIS:
```bash
cd /path/to/MatRIS
pip install -e .
```

### Install DeepMD-MatRIS

```bash
cd /path/to/deepmd-matris
pip install -e .
```

## Quick Start

### 1. Prepare Training Data

Prepare your training data in DeePMD-kit format (same as for other DeePMD models):

### 2. Configuration

Create an input JSON file (see `examples/water/matris/input.json`):

```json
{
  "model": {
    "type": "matris",
    "type_map": ["O", "H"],
    "pairwise_cutoff": 6.0,
    "three_body_cutoff": 4.5,
    "sel": "auto",
    "num_layers": 10,
    "is_intensive": true,
    "is_conservation": true
  }
}
```

### 3. Training

Train the model using DeePMD-kit's training script:

```bash
dp --pt train input.json
```

## Examples

See the `examples/` directory for complete training examples:

- `examples/water/matris/`: Water molecule training with O and H atoms

## Features

- **Graph Neural Network**: MatRIS uses atom graph and line graph for message passing
- **Conservative Forces**: Automatic differentiation for physically consistent forces and stress
- **Three-body Interactions**: Explicit three-body terms for better accuracy
- **DeePMD Integration**: Seamless integration with DeePMD-kit training and inference

## License

This project is licensed under the LGPL-3.0 license, consistent with both DeePMD-kit and MatRIS.

## Acknowledgments

This plugin builds upon the excellent work of:
- The MatRIS team
- The DeePMD-kit team at Deep Modeling
- The PyTorch Geometric developers 
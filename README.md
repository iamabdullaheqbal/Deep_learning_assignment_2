# Deep Learning Assignment 2: Neural Network from Scratch

**Course:** Deep Learning and Neural Networks  
**Student Name:** Abdullah  
**Roll Number:** 54  
**Submission Date:** December 5, 2025

## Overview

This assignment implements a 2-layer neural network for binary classification from scratch using only NumPy. The implementation includes forward propagation, backward propagation, and gradient descent optimization.

## Assignment Requirements

Implement the following components:

1. **Parameter Initialization** (`Init_parameters`)
   - Initialize weights and biases for a 2-layer network
   - Parameters: N_x (input features), N_h (hidden units), N_y (output units)

2. **Activation Functions**
   - ReLU activation for hidden layer
   - Sigmoid activation for output layer

3. **Forward Propagation** (`forward_pass`)
   - Compute predictions using current parameters
   - Return predictions and intermediate values for backpropagation

4. **Cost Function** (`compute_cost`)
   - Calculate cross-entropy loss

5. **Backward Propagation** (`backward_pass`)
   - Compute gradients for all parameters

6. **Training Function** (`fit`)
   - Implement gradient descent
   - Update parameters over multiple epochs

## Dataset

The implementation uses the **Boston Housing Dataset** for binary classification:
- Target variable: Median house value (medv) converted to binary (above/below median)
- Features: 13 housing-related attributes
- Split: 80% training, 20% testing

## Project Structure

```
.
├── abdullah_54.ipynb          # Main implementation notebook
├── dataset/
│   └── BostonHousing.csv     # Dataset file
├── pyproject.toml            # Project dependencies
├── uv.lock                   # Lock file
└── README.md                 # This file
```

## Setup Instructions

This project uses [uv](https://github.com/astral-sh/uv) as the package manager.

### Prerequisites

- Python 3.8+
- uv package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/iamabdullaheqbal/Deep_learning_assignment_2
cd Deep_learning_assignment_2
```

2. Install uv (if not already installed):
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

3. Create virtual environment and install dependencies:
```bash
uv sync
```

4. Activate the virtual environment:
```bash
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

## Running the Notebook

1. Start Jupyter:
```bash
jupyter notebook abdullah_54.ipynb
```

Or use VS Code with Jupyter extension.

2. Run all cells to:
   - Load and preprocess the dataset
   - Train neural networks with different hidden layer sizes
   - Evaluate performance on train and test sets
   - Visualize results

## Implementation Details

### Network Architecture

- **Input Layer:** 13 features (housing attributes)
- **Hidden Layer:** Variable size (tested with 4, 8, 16 units)
- **Output Layer:** 1 unit (binary classification)

### Hyperparameters

- Learning rate: 0.01
- Iterations: 2000
- Activation functions:
  - Hidden layer: ReLU
  - Output layer: Sigmoid

### Results

The notebook tests three different hidden layer configurations:
- 4 hidden units
- 8 hidden units
- 16 hidden units

Performance metrics (accuracy) are reported for both training and test sets.

## Dependencies

- numpy: Numerical computations
- pandas: Data manipulation
- scikit-learn: Data preprocessing and evaluation metrics
- matplotlib: Visualization

## Key Features

✅ **From Scratch Implementation:** No high-level ML libraries used for the neural network  
✅ **Proper Gradient Descent:** Manual implementation of backpropagation  
✅ **Multiple Configurations:** Tests different network architectures  
✅ **Visualization:** Loss curves and performance metrics  
✅ **Clean Code:** Well-documented with clear function definitions

## Notes

- This is a pure NumPy implementation as required by the assignment
- The network is trained from scratch without using TensorFlow, PyTorch, or Keras
- StandardScaler is used only for feature normalization (preprocessing)
- Evaluation metrics from sklearn are used only for measuring performance

## License

This project is submitted as part of academic coursework.

## Contact

**Abdullah**  
Roll No: 54  
Course: Deep Learning and Neural Networks

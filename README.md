# Deep Learning Repository

This repository contains a collection of mini-projects showcasing various machine learning and deep learning concepts studied during my coursework. Each project demonstrates specific techniques and algorithms relevant to the field of deep learning.

## Project Overview

### 1. Potato Disease Classification

This project implements a Convolutional Neural Network (CNN) to classify potato plant diseases from images. The model can identify different diseases affecting potato plants, which is essential for early detection and treatment in agricultural settings.

**Key Features:**
- Image classification using TensorFlow/Keras
- Data augmentation techniques
- CNN architecture with multiple convolutional and pooling layers
- Model evaluation and performance visualization
- Image preprocessing and normalization

### 2. Gradient Descent Implementation

This project demonstrates the gradient descent optimization algorithm using a binary classification problem based on insurance data. It shows how the algorithm works to minimize the loss function.

**Key Features:**
- Implementation of gradient descent algorithm
- Binary classification using sigmoid activation
- Loss evaluation
- Data preprocessing and normalization
- 
### 3. Regularization Techniques

This project explores different regularization techniques to prevent overfitting in machine learning models, specifically focusing on logistic regression applied to insurance data.

**Key Features:**
- L1 and L2 regularization implementations
- Comparison of regularization effects on model coefficients
- Custom model creation with TensorFlow/Keras
- Data visualization

### 4. Cost and Loss Functions

Two implementations exploring various cost and loss functions used in machine learning:

#### 4.1 Cost Loss Function for Insurance Data
- Binary cross-entropy loss implementation
- Gradient descent optimization
- Mean Absolute Error (MAE) calculation
- Log loss implementation

#### 4.2 Loss Cost Function (Hardcoded)
- Manual implementation of common loss functions
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Log loss function with numerical stability considerations

### 5. HR Analytics

This project analyzes HR data to predict employee attrition using logistic regression. It includes data visualization to understand factors influencing employee retention.

**Key Features:**
- Employee attrition prediction
- Data visualization with correlation matrices and bar charts
- One-hot encoding for categorical variables
- Model evaluation using confusion matrix and classification report

## Getting Started

### Prerequisites
- Python 3.x
- TensorFlow 2.x
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kxtubh/Deep-Learning.git
```

2. Install the required packages:
```bash
pip install tensorflow scikit-learn numpy pandas matplotlib seaborn
```

### Running the Projects

Each project is contained in its own Python file. To run a specific project, use:
```bash
python <filename>.py
```

Replace `<filename>` with the name of the project file you want to run.

## Learning Outcomes

Through these projects, I've demonstrated understanding of:
- Neural network architectures including CNNs
- Optimization algorithms like gradient descent
- Regularization techniques for preventing overfitting
- Various loss and cost functions and their implementations
- Data preprocessing techniques for machine learning
- Model evaluation and visualization
- Application of machine learning to real-world problems

## Future Work

- Implement more complex deep learning architectures
- Explore natural language processing projects
- Add reinforcement learning examples
- Deploy models to web applications

## License

This project is licensed under the MIT License - see the LICENSE file for details.

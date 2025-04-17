# CIFAR-10 Image Classification with CNNs and VAEs

This project applies machine learning concepts using the CIFAR-10 dataset. The goal is to develop a supervised learning solution to classify images using Convolutional Neural Networks (CNNs) and explore an unsupervised learning approach using Variational Autoencoders (VAEs) for image reconstruction. Additionally, this project compares the performance and capabilities of CNNs and VAEs.

## Objectives
1. **Learning Problem**:
   - Develop a supervised learning solution to classify CIFAR-10 images using a CNN.
   - Explore an unsupervised learning approach with a Variational Autoencoder (VAE) for image reconstruction.
   - Compare performance and capabilities of CNNs and VAEs.

2. **Dataset**:
   - The CIFAR-10 dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/cifar10-python-in-csv?resource=download) and preprocessed for training.

3. **Project Requirements**:
   - Use a novel neural network architecture (e.g., VAE).
   - Investigate different design choices and compare their performance.
   - Train and validate models effectively while visualizing the results.

## Project Setup and Folder Structure
Before running the project, follow these steps to properly set up the data directory:

Download the CIFAR-10 CSV dataset
Go to the [Kaggle](https://www.kaggle.com/datasets/fedesoriano/cifar10-python-in-csv?resource=download) CIFAR-10 CSV dataset and download the dataset.

Create the following folder structure in your repo root:

``` bash
cifar10/
├── train.csv
├── test.csv
└── images/
    ├── frog.jpg
    ├── airplane.jpg
    └── ... (any test images you want to run through the model)
```
Place the downloaded CSV files (train.csv, test.csv) inside the `cifar10/` folder.

Place any custom or example images for prediction inside the `cifar10/images/` folder.

## Step 1: Setup and Installation

To install the necessary dependencies, you need to install the following Python libraries:
- `torch` for PyTorch
- `torchvision` for dataset handling and transformations
- `matplotlib` for visualizations
- `pandas` for data manipulation
- `scikit-learn` for dataset splitting

## Step 2: Load and Preprocess the Dataset

- The CIFAR-10 dataset is loaded and preprocessed, including splitting into training and validation sets.
- Images are normalized to have pixel values between 0 and 1 for efficient model training.
- A custom dataset class is created to handle the CIFAR-10 data using PyTorch’s `DataLoader` for batching.

## Step 3: Define the Models

### Model 1: Simple CNN
- A simple CNN is created with two convolutional layers followed by fully connected layers. The model is designed to classify the CIFAR-10 images into 10 categories.

### Model 2: Variational Autoencoder (VAE)
- A VAE is a generative model designed to learn a latent space representation of the input data.
- It consists of an encoder to compress input images and a decoder to reconstruct them, aiming for a smooth, continuous latent space.

## Step 4: Define Training Components

- **Loss Functions**:
  - `CrossEntropyLoss` for CNN classification tasks.
  - `Mean Squared Error (MSE)` for VAE reconstruction tasks.
  
- **Optimizers**:
  - The Adam optimizer is used for both models to minimize the respective loss functions.

## Step 5: Training Loop

- The training loop trains both the CNN and VAE models for a specified number of epochs.
- For the CNN, the model is trained to minimize classification loss. 
- For the VAE, the model is trained to minimize the reconstruction loss (MSE) between the original and reconstructed images.

## Step 6: Compare Models

- After training, the models are evaluated:
  - **CNN**: The model’s performance is assessed using the training and validation losses, and its classification accuracy is reported.
  - **VAE**: The reconstructed images from the VAE are compared to the original images to evaluate the model's ability to learn useful latent representations.

## Conclusion

This project demonstrates:
1. The effectiveness of CNNs for supervised classification tasks on the CIFAR-10 dataset.
2. The potential of VAEs for unsupervised learning, particularly for image reconstruction.
3. A comprehensive comparison between CNNs and VAEs, highlighting their respective strengths and limitations.

By experimenting with different architectures and training techniques, the project offers insights into both supervised and unsupervised deep learning approaches.

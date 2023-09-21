# Project Title: Neural Network and Autoencoder Implementation for Data Classification and Dimensionality Reduction

## Neural Network Warm-up Problem:
### 1. Data Generation and Preprocessing
- Generated synthetic data points following complex rules.
- Created a training dataset with 500 samples, a validation dataset with 500 samples, and a test dataset with 1,000 samples.
- Calculated the relative frequency of each class in the training data.

### 2. Neural Network Training
- Implemented a neural network with one hidden layer to classify the data.
- Utilized the validation dataset for hyperparameter tuning, including learning rate, hidden layer size, and regularization.
- Explored different options for model selection and documented the chosen model.

### 3. Evaluation and Analysis
- Reported the overall accuracy of the classifier on the test data.
- Generated two plots showing true labels and predicted labels on a grid, providing insights into model performance.

## Autoencoder:
### 1. Data Preprocessing
- Utilized images of galaxies from the Galaxy Zoo project.
- Resized and converted images to 20x20 pixels and grayscale.

### 2. PCA Dimensionality Reduction
- Employed PCA to reduce images to 25 dimensions and reconstructed them.
- Calculated the reconstruction error on the validation and test sets.

### 3. Autoencoder Training
- Trained an autoencoder with a 25-dimensional bottleneck layer.
- Explored various hyperparameters, such as activation functions, learning rates, regularization, and hidden layer sizes.
- Measured model performance using the average squared error per pixel on the validation set.

### 4. Comparison with PCA
- Compared autoencoder and PCA performance using the test data.
- Analyzed situations where one method outperformed the other.

### 5. Image Reconstruction
- Displayed original images, PCA reconstructions, and autoencoder outputs side-by-side.
- Commented on the suitability of each technique for different image qualities.

### 6. Finding Similar Images
- Identified the three closest training set images to selected test images using PCA and autoencoder representations.
- Provided insights into the effectiveness of both approaches for finding similar images.

## Key Achievements:
- Successfully implemented neural networks and autoencoders for data classification and dimensionality reduction.
- Conducted data preprocessing, model training, and performance evaluation.
- Explored hyperparameters and documented model selection processes.
- Compared autoencoder performance with PCA for dimensionality reduction.
- Demonstrated the ability to analyze image reconstructions and find similar images using different techniques.

## Skills Demonstrated:
- Neural Network Implementation
- Hyperparameter Tuning
- Data Preprocessing
- Model Evaluation
- Dimensionality Reduction (PCA and Autoencoder)
- Image Processing and Analysis

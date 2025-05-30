# NOAA Fisheries Steller Sea Lion Population Count

## Project Overview

This project aims to automatically identify and count sea lions in aerial images using computer vision and deep learning techniques. The project utilizes the NOAA Steller Sea Lion Population Count dataset from a Kaggle competition.

## Key Features

- **Image Preprocessing**: Scaling, Gaussian blur, and other preprocessing operations on original aerial images
- **Sea Lion Detection**: Using blob detection algorithms to identify sea lion locations in images
- **Color Classification**: Classifying sea lions based on colored dots in annotated images (red, magenta, green, blue, brown)
- **Deep Learning Model**: Building convolutional neural networks for sea lion count prediction

## Technology Stack

- **Python 3.11**
- **OpenCV**: Image processing and computer vision
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization
- **scikit-image**: Image processing algorithms
- **Keras**: Deep learning framework
- **Jupyter Notebook**: Development environment

## Dataset Structure

```
KaggleNOAASeaLions/
├── Train/              # Original training images
├── TrainDotted/        # Training images with annotation dots
└── MismatchedTrainImages.txt  # List of mismatched images
```

## Installation Requirements

```bash
pip install numpy opencv-python matplotlib scikit-image keras tensorflow
```

## Usage Instructions

1. **Data Preparation**
   ```python
   # Set image scaling ratio and patch size
   r = 0.4  # Scaling ratio
   width = 100  # Patch size
   ```

2. **Run Main Program**
   ```bash
   jupyter notebook main.ipynb
   ```

3. **Model Training**
   - Load and preprocess image data
   - Use blob detection algorithms to identify sea lion locations
   - Classify sea lion types based on colors
   - Train convolutional neural network model

## Core Algorithms

### Image Processing Pipeline

1. **Image Difference Calculation**: Calculate absolute difference between original and annotated images
2. **Mask Creation**: Create binary masks to highlight sea lion regions
3. **Blob Detection**: Use LoG (Laplacian of Gaussian) blob detector to identify sea lions

### Color Classification Logic

```python
# Sea lion classification based on RGB values
if R > 225 and b  225 and b > 225 and g < 25:  # Magenta
    res[x1,y1,1] += 1
elif R < 75 and b < 50 and 150 < g < 200:  # Green
    res[x1,y1,4] += 1
elif R < 75 and 150 < b < 200 and g < 75:  # Blue
    res[x1,y1,3] += 1
elif 60 < R < 120 and b < 50 and g < 75:   # Brown
    res[x1,y1,2] += 1
```

## Model Architecture

Convolutional Neural Network built with Keras, including:
- **Convolutional Layers** (Conv2D): Feature extraction
- **Pooling Layers** (MaxPooling2D): Dimensionality reduction and feature selection
- **Dense Layers**: Final classification and counting

## Evaluation Metrics

Uses Root Mean Square Error (RMSE) as the model performance evaluation metric:

```python
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
```

## Project Structure

```
project/
├── main.ipynb          # Main program file
├── README.md           # Project documentation
└── data/              # Data folder
    ├── Train/         # Original images
    └── TrainDotted/   # Annotated images
```
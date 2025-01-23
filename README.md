# Simple AI Image Classifier using TensorFlow

## Project Overview

This project demonstrates how to build a simple Convolutional Neural Network (CNN) for image classification using TensorFlow and Keras. The goal of this project is to classify images into different categories, such as "cats" vs. "dogs," based on a given dataset of images. The model is trained to learn patterns and features from labeled image data, and can be used to predict categories for new, unseen images.

## Features

- Build a CNN to classify images into two categories (binary classification).
- Use of data augmentation techniques (such as rotation, zoom, and flipping) to improve model generalization.
- Visualize the model's performance by plotting training and validation accuracy/loss curves.
- Ability to classify new images using the trained model.

## Requirements

Before running the project, make sure you have the following Python libraries installed:

- `tensorflow==2.11.0`
- `numpy==1.23.5`
- `matplotlib==3.6.3`

You can install them using the `requirements.txt` file.

### To Install Dependencies:

```
pip install -r requirements.txt
```

## Dataset

The dataset used for training the model should be organized into two main directories:

- **Train Directory** (`train_data/`): Contains subdirectories for each class (e.g., `cats/`, `dogs/`).
- **Validation Directory** (`validation_data/`): Contains subdirectories for each class for validation.

Each class folder contains images of that specific class. Make sure the images are in a format supported by TensorFlow (JPEG, PNG, etc.).

## Directory Structure

```
Simple-AI-Image-Classifier/
│
├── train_data/
│   ├── cats/
│   └── dogs/
│
├── validation_data/
│   ├── cats/
│   └── dogs/
│
├── model.py               # Main code for building, training, and evaluating the model
├── requirements.txt       # Required libraries
├── README.md              # Project documentation
└── image_classifier_model.h5  # Saved model after training (if applicable)
```

## How to Use the Project

### 1. Prepare Your Dataset

Organize your dataset into directories as described above, with each class in its own folder. You should have a training set and a validation set.

### 2. Modify the Code for Your Dataset

In the `model.py` script, you need to modify the following paths to point to your dataset:

```python
train_dir = 'path_to_train_data'  # Path to your training data directory
validation_dir = 'path_to_validation_data'  # Path to your validation data directory
```

Make sure the directory structure matches the one described above.

### 3. Run the Script

After updating the paths, you can run the `model.py` script to start the training process.

```bash
python model.py
```

The script will:

1. Preprocess the images and apply data augmentation techniques.
2. Build the CNN model.
3. Train the model using the images from your `train_data/` folder.
4. Evaluate the model on the validation data in the `validation_data/` folder.
5. Plot training and validation accuracy/loss curves.
6. Save the trained model as `image_classifier_model.h5`.

### 4. Test the Model with New Images

Once the model is trained, you can use it to classify new images. Simply use the `classify_image()` function to pass the path of an image you want to test.

```python
image_path = 'path_to_test_image.jpg'
predictions = classify_image(image_path)
print(f"Prediction: {predictions[0][0]}")
```

### 5. Visualize Performance

After training, the script will automatically generate two plots:

- **Model Accuracy**: Shows how accuracy changes over epochs for both training and validation datasets.
- **Model Loss**: Shows how loss changes over epochs for both training and validation datasets.

## Model Architecture

The model uses a Convolutional Neural Network (CNN) architecture:

1. **Conv2D Layer**: Applies a set of filters to extract features from the image.
2. **MaxPooling2D Layer**: Reduces the spatial dimensions of the feature map.
3. **Flatten Layer**: Converts the 2D feature map to a 1D vector for the fully connected layers.
4. **Dense Layer**: Fully connected layers to make predictions.
5. **Activation Function**: A `sigmoid` function is used for binary classification (change to `softmax` for multi-class).

## Training the Model

The model is trained using the `Adam` optimizer and `binary_crossentropy` loss function. The number of epochs and batch size can be adjusted in the script:

```python
epochs = 10
batch_size = 32
```

Feel free to experiment with different hyperparameters to improve the performance of the model.

## Saving and Loading the Model

After training, the model is saved as `image_classifier_model.h5`. You can load this model later for inference or further fine-tuning using:

```python
from tensorflow.keras.models import load_model
model = load_model('image_classifier_model.h5')
```

## Contributions

Feel free to contribute to this project by adding new features, fixing bugs, or improving the model's performance. Open a pull request to get your changes merged.

## License

This project is open-source and available under the [MIT License](LICENSE).

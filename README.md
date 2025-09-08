# Project Documentation

## Overview

This project focuses on Building CNN model to classify images of 90 different classes of animals. The core components include data preprocessing, model training, evaluation, and experimentation, primarily organized within the `Data`, `models`, and `notebook` directories. The project is structured to facilitate reproducible research and modular development of data-driven solutions.

---

## Directory Structure

### Data

- **Purpose:** Contains raw and processed datasets used throughout the project.
- **Contents:** 
    - 90 folders each of different class and containing 60 images in each folder

### models

- **Purpose:** 
    - Saved the models one is feature extraction model and other is fine-tuned model
- **Contents:**
    - Model `classifier_2_acc_85.h5` is a fine-tuned model which got 85+ accuracy on test data
    - Model `classifier_model_1.h5` is a simple feature extraction model which got 75 accuracy

### notebook

- **Purpose:** Contains Jupyter notebooks for exploratory data analysis, prototyping, and documentation of experiments.
- **Contents:**
    - `CNN.ipynb` is main notebook used for development of the model
    - `checking.ipynb` is used for checking the saved model

---

## Getting Started

1. **Data Preparation:** Begin by exploring the `Data` directory to understand the available datasets and preprocessing steps.
2. **Model Development:** Use scripts in the `models` directory to train and evaluate machine learning models.
3. **Experimentation:** Leverage the `notebook` directory for interactive analysis and prototyping.

---

## Using the Model of local system

- Go to `models` folder and copy the path of required model.
- Open a new jupyter notebook and import the dataset 
```
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "..\\Data\\animals\\", <=== #replace it your directory
    batch_size=32, 
    image_size=(224,224),
    shuffle=True,
    color_mode="rgb" 
)
```
- Load the model using `load_model` module from `tensorflow.keras.models`
```
from tensorflow.keras.models import load_model
```
- ## Coverting the image to vector and preprocessing
    - Convert the image to vector using
    ```
    vector = image.img_to_array(img)
    ```
    - To visualize the image use `matplotlib`
    ```
    plt.imshow(vector.astype("uint8"))
    ```
    - Apply preprocessing and prediction
    ```
    from tensorflow.keras.applications.vgg16 import preprocess_input
    
    vector = preprocess_input(x)

    preds = model.predict(x)
    ```
    - `preds` will return an array with probabilities
    ```
    pred_class = np.argmax(preds[0]) ## returns the highest probabiliy from the array
    print("Predicted class:", class_names[pred_class])
    ```
---
## Author
    - Name - `Sjunaid`
    - Email - `sjunaid2034@gmail.com`
    - GitHub - https://github.com/gyrfalcon55
---

# Thank You!
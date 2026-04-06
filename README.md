# CVI-Miletsone
CVI MILESTONE
## Project Overview
This project was created for the CVI620 Final Project.  
The goal of the project is to train a convolutional neural network (CNN) model that can predict the steering angle of a self-driving car by using images captured from the simulator's front camera.

The final trained model is tested in the Udacity self-driving car simulator in Autonomous Mode.

---

## Project Files
This project includes the following main files:

- `data_preprocessing.py`  
  Loads the driving data, fixes image paths, checks the steering distribution, and balances the dataset.

- `utils.py`  
  Contains image preprocessing, data augmentation, image reading, and batch generator functions.

- `train_mode.py`  
  Builds and trains the CNN model, saves the final model, and plots the training history.

- `TestSimulation.py`  
  Loads the trained model and connects it to the simulator for autonomous driving.

- `model.h5`  
  Final trained model file.

- `training_history.png`  
  Training and validation loss graph.

- `collected_data/IMG`  
  Contains the recorded simulator images.

- `collected_data/driving_log.csv`  
  Contains the recorded driving data and steering values.

---

## Approach
The project was completed in the following steps:

1. Data was collected manually in Training Mode using the Udacity simulator.
2. The steering angle distribution was checked using a histogram.
3. The dataset was balanced by reducing the number of near-zero steering samples.
4. Images were preprocessed before training.
5. Random augmentation was applied during training to improve generalization.
6. A CNN model based on the Nvidia end-to-end driving idea was trained.
7. The trained model was tested in Autonomous Mode inside the simulator.

---

## Data Collection
The data was collected manually by driving the car in the simulator.

The collected dataset includes:
- center camera images
- steering angle values

The data was saved automatically by the simulator into:
- `IMG` folder
- `driving_log.csv`

---

## Data Preprocessing
The following preprocessing steps were applied:

- Cropping the road area from the image
- Converting the image from RGB to YUV
- Applying Gaussian Blur
- Resizing the image to `200 x 66`
- Normalizing pixel values by dividing by `255.0`

These steps were used to make the input more suitable for the CNN model.

---

## Data Augmentation
To improve model performance, random augmentation was used during training:

- Horizontal flipping
- Brightness adjustment
- Zoom
- Pan
- Rotation

For some transformations, the steering angle was also adjusted so that the image and label stayed consistent.

---

## Model Architecture
The model is a CNN built using TensorFlow / Keras.

It includes:
- 5 convolutional layers
- Flatten layer
- Fully connected dense layers
- Dropout layer
- Final output layer with 1 value for steering angle prediction

The model was trained using:
- `Adam` optimizer
- `Mean Squared Error (MSE)` loss

---

## Training
The model was trained on the collected and balanced dataset.

Training settings:
- Batch size: `32`
- Epochs: `15`
- Validation split: `20%`

After training:
- the model was saved as `model.h5`
- the loss graph was saved as `training_history.png`

---

## Challenges Faced
Some challenges were faced during this project:

1. **Simulator compatibility problem**  
   The first simulator version did not run correctly on the system, so another compatible version was used.

2. **Environment setup issues**  
   The original package list was outdated, so a working environment was created manually with compatible package versions.

3. **Model loading issue**  
   The saved model caused a loading error during testing, so the model was loaded with `compile=False`.

4. **Weak driving performance at first**  
   At the beginning, the car could not stay on the track well. This was improved by:
   - stronger balancing
   - better augmentation logic
   - lower speed in testing
   - scaling steering predictions

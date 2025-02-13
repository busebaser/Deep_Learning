# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 17:40:14 2025

Creating an Artificial Neural Network model using the MNIST dataset.

@author: Buse
"""
# %% Dataset Preparation and Preprocessing
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense 

from keras.models import load_model 

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load the MNIST dataset and separate it into training and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Visualizing a few examples
plt.figure(figsize=(10,5))

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_train[i],cmap="gray")
    plt.title(f"index : {i}, Label: {y_train[i]}")
    plt.axis("off")
plt.show()

# Normalize the dataset by scaling pixel values from the range 0-255 to 0-1
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]*x_train.shape[2])).astype("float32")/255
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]*x_test.shape[2])).astype("float32")/255

# Convert labels to categorical format using one-hot encoding
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

# %% Creating and compiling the ANN model

model = Sequential()

# First Layer : 512 cell, Relu Activation function, input size = 28*28 = 784
model.add(Dense(512, activation="relu",input_shape=(28*28,)))
# Second Layer : 256 cell, tanh Activation function
model.add(Dense(256, activation="tanh"))
# Output Layer: Must have 10 units since there are 10 classes, and softmax activation is required for multi-class classification
model.add(Dense(10, activation="softmax"))

model.summary()

# Model Compilation:
# Optimizer: Adam - Ideal for large datasets and complex networks
# Loss: Categorical Crossentropy - Suitable for multi-class classification
# Metric: Accuracy - To evaluate model performance

model.compile(optimizer="adam", 
              loss="categorical_crossentropy",
              metrics=["accuracy"])


# %% Defining Callbacks and Training the ANN

# Early stopping is applied if validation loss does not improve
# Monitor: Observes the validation loss
# Patience: 3 -> Stops training if validation loss does not improve for 3 epochs
# restore_best_weights: Restores the best model weights

early_stopping = EarlyStopping(monitor= "val_loss",patience=3,restore_best_weights=True)

# save_best_only = Saves only the best-performing model
checkpoint = ModelCheckpoint("ann_best_model.keras", monitor="val_loss", save_best_only=True)

history = model.fit(x_train, y_train,
          epochs=10,
          batch_size=60,
          validation_split=0.2,
          callbacks=[early_stopping, checkpoint])

# %% Model Evaluation, Visualization, Model Save and Load

# Evaluating model performance using test data
# Evaluate: Computes test loss (test_loss) and test accuracy (test_acc) on the test dataset

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test acc : {test_acc}, test loss : {test_loss}")

# Visualizing training and validation accuracy
plt.figure()
plt.plot(history.history["accuracy"], marker = "o",label = "Training Accuracy")
plt.plot(history.history["val_accuracy"],  marker = "o",label = "Validation Accuracy")
plt.title("ANN Accuracy on MNIST Data Set")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Visualizing training and validation loss
plt.figure()
plt.plot(history.history["loss"], marker = "o", label="Training Loss")
plt.plot(history.history["val_loss"], marker = "o", label="Validation Loss")
plt.title("ANN Loss on MNIST Data Set")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


model.save("final_mnist_ann_model.h5")

loaded_model = load_model("final_mnist_ann_model.h5")
test_loss, test_acc = loaded_model.evaluate(x_test, y_test)
print(f"Loaded Model Result -> Test acc : {test_acc}, test loss : {test_loss}")


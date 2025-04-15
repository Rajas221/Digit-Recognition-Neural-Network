# Step 1: Import Required Libraries


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical


#2: Load and Preprocess the MNIST Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train has 60k training images and x_test has 10k testing images and y_train and y_test are the corresponding labels.


#Visualise a sample image
plt.imshow(x_train[1], cmap='gray')
plt.title(f"Label: {y_train[1]}")
plt.show()


#Preprocess the data
x_train = x_train/ 255.0
x_test = x_test/ 255.0
#Pixel Values range from 0-255, so we divide by 255 to get values between 0 and 1 for faster and better training


#Convert Labels to one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#One hot encoding transforms a categorical label into a binary verctor where all elements aer 0 excpt the index of the label which is 1.
#For example, if the label is 3, it will be converted to [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#This is done to make the model output a probability distribution over the classes.
#This is useful for multi-class classification problems.



#3: Build the Neural Network Model
model = Sequential()  # Create a sequential model
#Sequential model is a linear stack of layers. It allows you to build a model layer by layer.
model.add(Flatten (input_shape=(28, 28))),  # Flatten the 28x28 images into a 784-dimensional vector
model.add(Dense(128, activation='relu')),  # Hidden layer with 128 neurons and ReLU activation
model.add(Dense(64, activation='relu')),  # Hidden layer with 64 neurons and ReLU activation #ReLu is a non-linear activation function that helps the model learn complex patterns in the data.
#ReLU is defined as f(x) = max(0, x). It outputs 0 for negative inputs and the input itself for positive inputs.
#This helps the model learn complex patterns in the data by introducing non-linearity.
model.add(Dense(10, activation='softmax'))  # Output layer with 10 neurons (one for each digit) and softmax activation



#4 Model Compilation
model.compile(optimizer='adam', # Adam optimizer is an adaptive learning rate optimization algorithm that is designed to be efficient and effective.
              loss='categorical_crossentropy', # Categorical crossentropy loss function is used for multi-class classification problems. It measures the dissimilarity between the true labels and the predicted labels.
              metrics=['accuracy']) 



#5 Model Training
history = model.fit(x_train,y_train,
                    epochs=10, # Number of epochs to train the model. An epoch is one complete pass through the training data.
                    batch_size=32, # Number of samples per gradient update. The model is trained on a batch of samples at a time.
                    validation_split=0.1) # Fraction of the training data to be used as validation data. The model is evaluated on this data after each epoch.



#6 Evaluate on Test Data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%") 



#7 Predictions
predictions = model.predict(x_test) # Predict the labels for the test data
predicted_labels = np.argmax(predictions, axis=1) # Get the index of the highest probability for each sample
# This gives us the predicted labels for the test data
#argmax returns the indices of the maximum values along an axis. In this case, we are getting the index of the maximum value along axis 1 (the columns) for each row (sample).



#Visualizations
plt.imshow(x_test[0], cmap='gray') # Visualize the first test image
plt.title(f"Predicted Label: {predicted_labels[0]}, True Label: {y_test[0].argmax()}") # Display the predicted and true labels
plt.show() # Show the image

import tensorflow as tf
import numpy as np
import os
import cv2

TRAINING_DATA_PATH = "train"
# Read Data from TRAINING_DATA_PATH
def read_training_data():
    training_data = []
    training_labels = [] 
    for img in os.listdir(TRAINING_DATA_PATH):
        training_labels.append(0) if img.split('.')[0] == "dog" else training_labels.append(1)  # Add a training label where 0 == dog and 1 == cat
        image_data = cv2.imread(os.path.join(TRAINING_DATA_PATH, img), cv2.IMREAD_GRAYSCALE)    # Read image as grayscale
        image_data = cv2.resize(image_data, (300, 300))                                         # Resize image to 300x300
        image_data = np.asarray(image_data).reshape((300, 300, 1))                              # Resize numpy array to 300x300x1 to meet requirements for Conv2D layer
        training_data.append(image_data)                                                        # Add to training data
    training_labels = np.asarray(training_labels)                                               # Convert both training_data and training_labels to numpy arrays
    training_data = np.asarray(training_data)
    return training_data, training_labels

# Avoid reading the data if you already have done it. ALOT FASTER
if os.path.exists("training_data.npy") and os.path.exists("training_labels.npy"):
    print("LOADING DATA FROM NPY FILES")
    training_data = np.load("training_data.npy")
    training_labels = np.load("training_labels.npy")
else:    
    print("LOADING DATA FROM COLLECTION")
    training_data, training_labels = read_training_data()
    training_data = training_data / 255                     # Divide by 255 to normalize Grayscale data

    from sklearn.utils import shuffle   

    # shuffle data for training
    training_data, training_labels = shuffle(training_data, training_labels, random_state=0)

    # save data in npy files for later use.
    np.save("training_data", training_data)
    np.save("training_labels", training_labels)


# ONLY UNCOMMENT IF YOU ARE USING A GPU
'''physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)'''

# Clear up any things in still in the background in keras
tf.keras.backend.clear_session()

"""
    Deep Learning CNN model.
    Uses 2 2D-Convolutional Layers each with kernels of 3x3 and relu activation functions (because grayscale data cant be negative)
    MaxPooling layers to speed up calculations and focus more attention on aggregated features.

    The flatten layer then flattens input 
"""

class dog_cat_model(tf.keras.Model):
    
    def __init__(self):
        super(dog_cat_model, self).__init__()
        # By only putting in an input shape of 3 dimensions there is not batch dimension so the model will learn stochastically
        self.conv_1_layer = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(300, 300, 1))
        self.pool1 = tf.keras.layers.MaxPool2D(2)
        self.conv_2_layer = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPool2D(2)
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.4)
        self.dense = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.4)
        self.dense2 = tf.keras.layers.Dense(512, activation='relu')
        self.output_layer = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, input_data):
        x = self.conv_1_layer(input_data)
        x = self.pool1(x)
        x = self.conv_2_layer(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return self.output_layer(x)

# Initialize the model with the adam optimization function, a sparse categorical crossentropy loss function, and a metric of accuracy.
model = dog_cat_model()
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
# Train the model for 5 epochs
model.fit(training_data, training_labels, epochs=5)

# Change this path for your desired path
TESTING_DATA_PATH = "test1"
def get_training_data(data_num=1):
    image = []  # Initialize array
    image_data = cv2.imread(os.path.join(TESTING_DATA_PATH, str(data_num)+'.jpg'), cv2.IMREAD_GRAYSCALE) # Read image as grayscal
    image_data = cv2.resize(image_data, (300, 300))                                                      # Resize Image to specific image size 
    image_data = np.asarray(image_data).reshape((300, 300, 1))                                           # Reshape numpy matrix to accomadate for model   
    image.append(image_data)                                                                             # Append to the image array
    image = np.asarray(image)                                                                            # Convert to Numpy Array
    return image

# Ordered labeling for output
label = ['dog', 'cat']                            

import matplotlib.pyplot as plt

# Short method to just abstract this line, also uses the whole image for display
def get_image(x):
    return cv2.imread(os.path.join(TESTING_DATA_PATH, str(x)+'.jpg'))

# adjust figure size here for images    
plt.figure(figsize=(5,5))

# Plot 10 images with their expected labels.
for x in range(1,11):
    plot_label = label[np.argmax(model.predict(get_training_data(x)))]
    plt.plot(x)
    plt.title(plot_label)
    plt.imshow(get_image(x))
    plt.show()
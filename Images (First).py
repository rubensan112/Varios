
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

#import datasheet
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape) #Numero de muestras y tamaño en pixeles
print("Tamaño de la test set:")
print(test_images.shape)


#Preprocesing

#The data must be preprocessed before training the network. If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255:

'''
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
'''

#We scale these values to a range of 0 to 1 before feeding to the neural network model.
#For this, cast the datatype of the image components from an integer to a float, and divide by 255. Here's the function to preprocess the images:

train_images = train_images / 255.0
test_images = test_images / 255.0

'''
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

#plt.show()

'''
#Build the model

'''
Building the neural network requires configuring the layers of the model, then compiling the model.
The basic building block of a neural network is the layer. Layers extract representations from the data fed into them. And, hopefully, these representations are more meaningful for the problem at hand.

Most of deep learning consists of chaining together simple layers. Most layers, like tf.keras.layers.Dense, have parameters that are learned during training.


'''


model = keras.Sequential([
    #Transform the format of the images from a 2d-array 28 by 28, to 1d-array 28*28. This layer has no parameters to learn; it only reformats the data.
    keras.layers.Flatten(input_shape=(28, 28)),
    #After the pixels are flattened, the network consists of a sequence of two After the pixels are flattened, the network consists of a sequence of two
    keras.layers.Dense(128, activation=tf.nn.relu), #Este tiene
    keras.layers.Dense(10, activation=tf.nn.softmax) #Este contiene

    # Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes.
])



'''
Compile the model

Loss function —This measures how accurate the model is during training. We want to minimize this function to "steer" the model in the right direction.
Optimizer —This is how the model is updated based on the data it sees and its loss function.
Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.

'''

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

'''
Training the neural network model requires the following steps:

Feed the training data to the model—in this example, the train_images and train_labels arrays.
The model learns to associate images and labels.
We ask the model to make predictions about a test set—in this example, the test_images array. We verify that the predictions match the labels from the test_labels array.

'''

model.fit(train_images, train_labels, epochs=5) #model is 'fit' the training data


#Evaluamos la accuracy

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

'''
It turns out, the accuracy on the test dataset is a little less than the accuracy on the training dataset. This gap between training accuracy and test accuracy is an example of overfitting. 
Overfitting is when a machine learning model performs worse on new data than on their training data.

'''

predictions = model.predict(test_images)

print(predictions[0])

#A prediction is an array of 10 numbers.
# These describe the "confidence" of the model that the image corresponds to each of the 10 different articles of clothing.
# We can see which label has the highest confidence value

print(np.argmax(predictions[0]))
print(test_labels[0])


# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)

plt.show()


#Finally make a prediction about a single image

# Grab an image from the test dataset
img = test_images[0]
#tf.keras models are optimized to make predictions on a batch, or collection, of examples at once. So even though we're using a single image, we need to add it to a list:

img = (np.expand_dims(img,0))

predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

print(np.argmax(predictions_single[0]))


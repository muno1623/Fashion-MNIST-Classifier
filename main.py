import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow.keras.utils as np_utils
import tensorflow.keras.utils as np_utils
from tensorflow.keras.models import load_model
import pydot
import os
from os import listdir
from os.path import isfile, join
import sys
import shutil
import pickle
import struct
import numpy as np

def read_idx(filename):
    """Credit: https://gist.github.com/tylerneylon"""
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

x_train = read_idx("./fashion/train-images-idx3-ubyte")
y_train = read_idx("./fashion/train-labels-idx1-ubyte")
x_test = read_idx("./fashion/t10k-images-idx3-ubyte")
y_test = read_idx("./fashion/t10k-labels-idx1-ubyte")

# Training Parameters
batch_size = 128
epochs = 1

# Lets store the number of rows and columns
img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]

# Getting our date in the right 'shape' needed for Keras
# We need to add a 4th dimenion to our date thereby changing our
# Our original image shape of (60000,28,28) to (60000,28,28,1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# store the shape of a single image
input_shape = (img_rows, img_cols, 1)

# change our image type to float32 data type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize our data by changing the range from (0 to 255) to (0 to 1)
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Now we one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Let's count the number columns in our hot encoded matrix
print ("Number of Classes: " + str(y_test.shape[1]))

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]

# create model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = tf.keras.optimizers.Adadelta(),
              metrics = ['accuracy'])

print(model.summary())

# Checkpoint, Early Stopping, and Learning rates
checkpoint = ModelCheckpoint("E:/Computer Vision and Machine Learning Projects/BuildingCNN/Trained Models/Fruit_Classifier_Checkpoint.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', # value being monitored for improvement
                          min_delta = 0, #Abs value and is the min change required before we stop
                          patience = 3, #Number of epochs we wait before stopping
                          verbose = 1,
                          restore_best_weights = True) #keeps the best weigths once stopped

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience = 3, verbose = 1, min_delta = 0.001)

callbacks = [checkpoint, earlystop, reduce_lr]

history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data= (x_test, y_test), callbacks= callbacks)

loss_and_metrics = model.evaluate(x_test, y_test, verbose=0)
print("Loss Score", loss_and_metrics[0])
print("Accuracy Score", loss_and_metrics[1])
print(history)

#Visualizing Model
np_utils.plot_model(model, "E:/Computer Vision and Machine Learning Projects/BuildingCNN/model_plot.png",show_shapes=True, show_layer_names=True )

#saving model
model.save("E:/Computer Vision and Machine Learning Projects/BuildingCNN/fashionMNIST_5ep.h5")

#Visualizing Model
np_utils.plot_model(model, "E:/Computer Vision and Machine Learning Projects/BuildingCNN/model_plot.png",show_shapes=True, show_layer_names=True )

#Saving History of Model
pickle_out = open("model_history.pickle", "wb")
pickle.dump(history.history, pickle_out)
pickle_out.close()

#Loading history of model
pickle_in = open("model_history.pickle", "rb")
saved_history = pickle.load(pickle_in)
print(saved_history)

#plotting model
#Loss Plot
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
ep = range(1, len(loss_values)+1)
line1 = plt.plot(ep, val_loss_values, label = "Validation Loss")
line2 = plt.plot(ep, loss_values, label = "Training Loss")
plt.setp(line1, linewidth=2.0, marker="+", markersize=10.0)
plt.setp(line2, linewidth=2.0, marker="4", markersize=10.0)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()

#Accuracy Plot
loss_values = history_dict["accuracy"]
val_loss_values = history_dict["val_accuracy"]
ep = range(1, len(loss_values)+1)
line1 = plt.plot(ep, val_loss_values, label = "Validation Accuracy")
line2 = plt.plot(ep, loss_values, label = "Training Accuracy")
plt.setp(line1, linewidth=2.0, marker="+", markersize=10.0)
plt.setp(line2, linewidth=2.0, marker="4", markersize=10.0)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.show()

#Prdeiction
#loading model
classifier = load_model("E:/Computer Vision and Machine Learning Projects/BuildingCNN/fashionMNIST_5ep.h5")


def getLabel(input_class):
    number = int(input_class)
    if number == 0:
        return "T-shirt/top "
    if number == 1:
        return "Trouser"
    if number == 2:
        return "Pullover"
    if number == 3:
        return "Dress"
    if number == 4:
        return "Coat"
    if number == 5:
        return "Sandal"
    if number == 6:
        return "Shirt"
    if number == 7:
        return "Sneaker"
    if number == 8:
        return "Bag"
    if number == 9:
        return "Ankle boot"


def draw_test(name, pred, actual, input_im):
    BLACK = [0, 0, 0]

    res = getLabel(pred)
    actual = getLabel(actual)
    expanded_image = cv2.copyMakeBorder(input_im, 0, 0, 0, 4 * imageL.shape[0], cv2.BORDER_CONSTANT, value=BLACK)
    expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
    cv2.putText(expanded_image, "Predicted - " + str(res), (152, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
    cv2.putText(expanded_image, "   Actual - " + str(actual), (152, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255),
                1)
    cv2.imshow(name, expanded_image)


for i in range(0, 10):
    rand = np.random.randint(0, len(x_test))
    input_im = x_test[rand]
    actual = y_test[rand].argmax(axis=0)
    imageL = cv2.resize(input_im, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    input_im = input_im.reshape(1, 28, 28, 1)

    ## Get Prediction
    res = str(classifier.predict_classes(input_im, 1, verbose=0)[0])

    draw_test("Prediction", res, actual, imageL)
    cv2.waitKey(0)

cv2.destroyAllWindows()
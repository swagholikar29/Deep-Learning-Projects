import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Part 1 Code

# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Define the text labels
fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2 
                        "Dress",        # index 3 
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6 
                        "Sneaker",      # index 7 
                        "Bag",          # index 8 
                        "Ankle boot"]   # index 9

print(x_train.shape[0], 'train set')
print(x_test.shape[0], 'test set')

img_index = 5
label_index = y_train[img_index]
print ("y = " + str(label_index) + " " +(fashion_mnist_labels[label_index]))
plt.imshow(x_train[img_index])

# Normalize the data dimensions so that they are of approximately the same scale
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print("Number of train data - " + str(len(x_train)))
print("Number of test data - " + str(len(x_test)))

# Further break training data into train / validation sets (# put 5000 into validation set and keep remaining 55,000 for train)
(x_train, x_valid) = x_train[5000:], x_train[:5000] 
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# Reshape input data from (28, 28) to (28, 28, 1)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Print training set shape
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Print the number of training, validation, and test datasets
print(x_train.shape[0], 'train set')
print(x_valid.shape[0], 'validation set')
print(x_test.shape[0], 'test set')

#Part 1 model

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()

#Compile the model
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)

#Model Training
model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=10,
         validation_data=(x_valid, y_valid),
         callbacks=[checkpointer])

# Load the weights with the best validation accuracy
model.load_weights('model.weights.best.hdf5')

score = model.evaluate(x_test, y_test, verbose=0)

print('\n', 'Test accuracy:', score[1])

y_hat = model.predict(x_test)


#Part 2 TensorFlow Code

model_new = tf.keras.Sequential()

#Convolution layer with 64 filters, each 3x3, stride of 1 (i.e., apply the filter at all pixel locations),no padding
model_new.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='valid', input_shape=(28,28,1), strides=(1,1)))

#Max pool with a pooling width of 2x2, stride of 2, no padding
model_new.add(tf.keras.layers.MaxPooling2D(pool_size=2, padding='valid', strides=(2,2)))

#ReLU
model_new.add(tf.keras.layers.ReLU())

#Flatten the 64 feature maps into one long vector
model_new.add(tf.keras.layers.Flatten())

#Fully-connected layer to map into a 1024-dimensional vector and ReLU Activation Function 
model_new.add(tf.keras.layers.Dense(1024, activation='relu'))

#Fully-connected layer to map into a 10-dimensional vector and Softmax Activation Function
model_new.add(tf.keras.layers.Dense(10, activation='softmax'))

#Take a look at the model summary
model_new.summary()

#Compile the model
model_new.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])


checkpointer = ModelCheckpoint(filepath='model_new.weights.best.hdf5', verbose=2, save_best_only=True)

#Model Training
model_new.fit(x_train,
         y_train,
         batch_size=64,
         epochs=2,
         validation_data=(x_valid, y_valid),
         callbacks=[checkpointer])

# Load the weights with the best validation accuracy
model_new.load_weights('model_new.weights.best.hdf5')

yhat_new = model_new.predict(x_train[0:1,:,:,:])[0]

# Evaluate the model on test set
score_new = model_new.evaluate(x_test, y_test, verbose=0)

#Print test accuracy
print('\nTest accuracy of the new model:', score_new[1]*100,'%\n')



#Extracting the weights from the trained tensorflow model
def extractweights (model):
    image_size = 28
    filter_size = 3
    filters = 64

    W1 = np.zeros((filters*(image_size-2)*(image_size-2), image_size*image_size))
    b1 = np.zeros((filters*(image_size-2)*(image_size-2),))
    current_index = 0

    conv_weights = np.array(model.get_weights()[0])
    conv_bias = np.array(model.get_weights()[1])

    for row in range(image_size-2):
        for column in range(image_size-2):
            for filter in range(filters):
                for width in range(filter_size):
                    start_index = (row + width)*image_size + column
                    stop_index = start_index + filter_size
                    W1[current_index, start_index:stop_index] = conv_weights[width,:,0,filter]
                    b1[current_index] = conv_bias[filter]
                    
                current_index = current_index + 1
                
    b1 = np.atleast_2d(b1).T
    W2 = model.get_weights()[2].T
    b2 = model.get_weights()[3]
    b2 = np.atleast_2d(b2).T
    W3 = model.get_weights()[4].T
    b3 = model.get_weights()[5]
    b3 = np.atleast_2d(b3).T
    
    return W1, b1, W2, b2, W3, b3

#fully connected neural network function
def fc(W, b, x):
    return np.dot(W, x) + b

#maxpooling function
def maxpooling(x, poolingWidth):
    image_size = 28
    filters = 64
    r, c = poolingWidth[0], poolingWidth[1]
    output = np.zeros((13, 13, 64))
    x = x.reshape((image_size-2, image_size-2, filters))
    for row in range(0, image_size-2-1, r):
            for column in range(0, image_size-2-1, c):
                    for filter in range(filters):
                        tempX = x[row:row+2, column:column+2, filter]
                        output[row//2, column//2, filter] = np.nanmax(tempX)
    return output.flatten()

#Softmax function
def softmax(z):
    return np.exp(z)/np.sum(np.exp(z), axis=0, keepdims=True)

#ReLU Function
def relu (x):
    return(np.maximum(0,x))

#Accuracy Calculation
def accuracy(y_hat, y):
    y_hat = y_hat.T
    y = y.T

    Y_hat = np.argmax(y_hat,1)
    Y = np.argmax(y,1)
    accuracy = 100*np.sum(Y == Y_hat)/y.shape[0]

    return accuracy

#Load weights extracted from tensorflow model to the fully connected neural network
W1, b1, W2, b2, W3, b3 = extractweights(model_new)

x_random = np.reshape(x_train[0], (-1,1)) 
fc1 = fc(W1,b1, x_random)
mp1 = maxpooling(fc1, (2,2))
mp1a = relu(mp1)
f1 = np.atleast_2d(mp1a.flatten()).T
fc2 = fc(W2,b2, f1) 
fc2a = relu(fc2)
fc3 = fc(W3, b3, fc2a)
yhat_ = softmax(fc3) 
yhat2 = yhat_.reshape(-1)
#print("Yhat1 Shape", yhat_new.shape)
#print("Yhat2 Shape", yhat2.shape)
print("Prediction for new CNN Model \n\n", yhat_new, "\n")
print("Prediction for Converted Feed Forward Network \n\n", yhat2)


# As can be seen the outputs for the the new CNN model created with tensor flow and the feedforward model are the same.






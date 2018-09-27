import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import color
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from numpy.linalg import norm
from scipy.signal.signaltools import convolve2d
import os
import time

training_file   = "./traffic-signs-data/train.p"
validation_file = "./traffic-signs-data/valid.p"
testing_file    = "./traffic-signs-data/test.p"


with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# Number of training examples
n_train = X_train.shape[0]

# Number of validation examples
n_validation = X_valid.shape[0]

# Number of testing examples.
n_test = X_test.shape[0]

# What's the shape of an traffic sign image?
image_shape = X_train.shape[1:3]

# How many unique classes/labels there are in the dataset.
n_classes = np.unique(y_train).shape[0]

print("Original Number of training examples =", n_train)
print("Original Number of testing examples =", n_test)
print("Original Image data shape =", image_shape)
print("Original Number of classes =", n_classes)


import random
import cv2
### Normalize the data by subtracting the mean and dividing
### by stddev. Then shift so all are between 0 and 1.

def normalize(x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_min = (np.min(x) - x_mean) / x_std
    x_max = (np.max(x) - x_mean) / x_std
    x_norm = (((x.astype(np.float32) - x_mean) / x_std) - x_min) * 1/(x_max - x_min)
    #x_norm = ((x.astype(np.float32) - x_mean) / x_std)
    return x_norm

def convert_to_grayscale(x):
    gray = x * (0.2989, 0.5870, 0.1140)
    gray = np.sum(gray, axis = 3).reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    return gray

def lcn(x):
    h, w = x.shape[:2]
    k = np.ones((9,9))
    k /= 81
    meaned = convolve2d(x, k, mode = 'same')
    p = np.power(x, 2.0)

    s = convolve2d(p, np.ones((9,9)), mode = 'same')
    s = np.sqrt(s)

    m =  x - meaned
    lcned = (m/s)
    lcn_min = np.min(lcned)
    lcn_max = np.max(lcned)
    normed = (lcned - lcn_min) * (1/(lcn_max - lcn_min))
    return normed

def preprocess_images(x):
    #
    # Convert the original images to grayscale
    #
    x_gray = convert_to_grayscale(x)

    #
    # Normalize the data so all are
    # between 0 and 1
    #
    return normalize(x_gray)



def augment_dataset(x, y):
    sign_count = np.bincount(np.sort(y))
    number_to_augment = np.float32((np.max(sign_count * 2.5) - sign_count)/sign_count)
    number_to_augment_tally = np.zeros_like(number_to_augment)

    augmented = []
    augmentedLabels = []
    for i in range(len(x)):
        label = y[i]
        number_to_augment_tally[label] += number_to_augment[label]
        number_to_do = np.int8(number_to_augment_tally[label])
        number_to_augment_tally[label] -= number_to_do

        for j in range(number_to_do):
            img = x[i].copy()

            #
            # Scale the image. Use a random scaling
            # factor between .9 and 1.1
            #
            rows, cols = img.shape[:2]
            scale_factor = random.uniform(.9, 1.1)
            interpolation = cv2.INTER_LINEAR
            if scale_factor < 1.0:
                interpolation = cv2.INTER_AREA
            dst = cv2.resize(img, (int(scale_factor*cols), int(scale_factor*rows)), interpolation = interpolation)
            img = cv2.resize(dst, (img.shape[1], img.shape[0]), interpolation = interpolation)

            #
            # Translate the image a random amount
            # between -2 and 2 pixels in the x and
            # y directions
            #
            x_trans = random.uniform(-2, 2)
            y_trans = random.uniform(-2, 2)
            M = np.float32([[1, 0, x_trans],[0, 1, y_trans]])
            img = cv2.warpAffine(img,M,(cols,rows))

            #
            # Rotate the image. Use a random angle
            # betwee -15 and 15 degrees.
            #
            angle = random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), angle, 1)
            img = cv2.warpAffine(img, M, (cols,rows))

            #
            # Add the new image and its label to the augmented list
            #
            augmented.append(img)
            augmentedLabels.append(label)

    return augmented, augmentedLabels

print("Augmenting dataset...")
x_augmented, y_augmented = augment_dataset(X_train, y_train)
x_train = np.concatenate((X_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

augmented_count = np.bincount(np.sort(y_train))

print("Preprocessing dataset...")
x_train = preprocess_images(x_train)
x_valid = preprocess_images(X_valid)
x_test  = preprocess_images(X_test)

input_channels = x_train.shape[3]

print("Augmented Number of training examples =", x_train.shape[0])
print("Augmented Number of testing examples =", x_test.shape[0])
print("Augmented Image data shape =", x_train.shape)
print("Augmented Number of classes =", np.unique(y_train).shape[0])
print("Number of input channels = ", str(input_channels))
#
# Generate a label map
#
label_map = np.genfromtxt('signnames.csv', skip_header=1, dtype=[('int8'), ('S50')],  delimiter=',')

def label_lookup(x):
    if x >= 0 and x <= len(label_map):
        return label_map[x][1].decode("utf-8")
    else:
        return "UNKNOWN"

def weights_and_biases(shape):
    mu = 0
    sigma = 0.1
    w = tf.Variable(tf.truncated_normal(shape=shape, mean = mu, stddev = sigma))
    b = tf.Variable(tf.zeros(shape[-1]))
    return w, b

def conv2d(x, shape, stride = 1, padding='VALID'):
    weights, biases = weights_and_biases((shape))
    layer = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding)
    layer = tf.nn.bias_add(layer, biases)
    layer = tf.nn.relu(layer)
    return layer

def fc(x, shape):
    weights, biases = weights_and_biases((shape))
    fc_layer = tf.add(tf.matmul(x, weights), biases)
    return fc_layer

def csharpNet(x):
    #
    # First Convolutional layer
    # Input  = 32x32x1
    # Output = 30x30x32
    #
    conv1 = conv2d(x, (3, 3, 1, 32))
    print("conv1: ", conv1.shape)

    #
    # Second Convolutional layer
    # Input  = 30x30x32
    # Output = 28x28x60
    #
    conv2 = conv2d(conv1, (3, 3, 32, 60))
    print("conv2: ", conv2.shape)

    #
    # Max Pooling and Dropout
    # Input  = 28x28x60
    # Output = 14x14x60.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv2 = tf.nn.dropout(conv2, keep_prob = keep_prob)
    print("conv2: ", conv2.shape)

    #
    # Third Convolutional layer
    # Input  = 14x14x60
    # Output = 12x12x95
    #
    conv3 = conv2d(conv2, (3, 3, 60, 75))
    print("conv3: ", conv3.shape)

    # Fourth Convolutional layer
    # Input  = 6x6x95
    # Output = 4x4x128
    #
    conv4 = conv2d(conv3, (3, 3, 75, 100))
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv4 = tf.nn.dropout(conv4, keep_prob = keep_prob)
    print("conv4: ", conv4.shape)

    conv5 = conv2d(conv4, (3, 3, 100, 125))
    print("conv5: ", conv5.shape)

    #
    # Flatten.
    # Input  = 4x4x128
    # Output = 2048
    #
    flat = tf.contrib.layers.flatten(conv5)
    print("flat: ", flat.shape)

    #
    # Fully Connected 1:
    # Input  = 2048
    # Output = 1024
    #
    fc1 = fc(flat, (1125, 1024))
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob = keep_prob)
    print("fc1: ", fc1.shape)

    #
    # Fully Connected 2:
    # Input  = 1024
    # Output = 43
    #
    logits = fc(fc1, (1024, 43))
    print("logits: ", logits.shape)
    return logits

#
# TF Placeholders.
# - Keep_prob is used for dropout
# - x is the image training data
# - y is the image training data's labels
#
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, (None, 32, 32, input_channels))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)


EPOCHS        = 75
BATCH_SIZE    = 128
LEARNING_RATE = 1e-4

logits = csharpNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
training_operation = optimizer.minimize(loss_operation)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(x_train)

    print("Training...{} examples.".format(num_examples))
    total_time_start = time.time()
    for i in range(EPOCHS):
        epoch_start = time.time()
        x_train, y_train = shuffle(x_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

        training_accuracy = evaluate(x_train, y_train)
        validation_accuracy = evaluate(x_valid, y_valid)
        epoch_end = time.time()
        print("EPOCH {:2d} ({:2d} secs): {:.3f}  {:.3f}".format(i+1, int(epoch_end-epoch_start), training_accuracy, validation_accuracy))

    saver.save(sess, './csharpNet')
    print("Model saved")
    total_time_end = time.time()
    print("Total training time: " + int(total_time_end - total_time_start) + " seconds")


signs = []
labels_dict = {'german1.jpg':4, 'german2.jpg':14, 'german3.jpg':22, 'german4.jpg':25, 'german5.jpg':14, 'german6.jpg':33}
labels = []
supported_image_files = [ ".jpg", ".jpeg", ".png"]
for image_name in os.listdir('test_images/'):
    base, ext = os.path.splitext(image_name)
    if (ext.lower() in supported_image_files):
        img = cv2.imread('test_images/' + image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        signs.append(img)
        labels.append(labels_dict[image_name])

signs = np.array(signs, np.float32)
signs = preprocess_images(signs)

prediction=tf.argmax(logits,1)

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(signs, labels)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

# Samen Sethi 300157474
# Bowen Xue 300139793

# Download CIFAR-10
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tests as tests
import tarfile

cifar10_dataset_folder_path = 'cifar-10-batches-py'


class DeeplearnLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


if not isfile('cifar-10-python.tar.gz'):
    with DeeplearnLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            'cifar-10-python.tar.gz',
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open('cifar-10-python.tar.gz') as tar:
        tar.extractall()
        tar.close()

tests.test_folder_path(cifar10_dataset_folder_path)


get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

import explore

# Explore the dataset
batch_id = 2
sample_id = 5
explore.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)


# Normal the shape of image
# x: The image data list, and the shape of image is (32,32,3)
# return: Normal data in Numpy array
def normal(x):
    return x / 255


tests.test_normal(normal)


from keras.utils import np_utils


# Return each label's one hot encoded vector
# x: Sample lables list
# return: Numpy array of one-hot encoded labels
def one_hot_encode(x):
    return np_utils.to_categorical(x)


tests.test_one_hot_encode(one_hot_encode)

# (Training, Validation, and Testing Data) pre-process
explore.preprocess_and_save_data(cifar10_dataset_folder_path, normal, one_hot_encode)


import tests as tests
import explore
import pickle
import tensorflow as tf

# Load the Pre-processed Validation data
valid_features, valid_labels = pickle.load(open('pre-process_validation.p', mode='rb'))


# Return the tensor to input a batch of images
# image_shape: the image's shape
# return: The image input after Tensor
def neural_image_input(image_shape):
    shape = [None] + list(image_shape)
    return tf.placeholder(tf.float32, shape=shape, name="x")


# Return the tensor to input batches' label
# n_classes: Number of classes
# return: The label input after Tensor.
def neural_label_input(n_classes):

    return tf.placeholder(tf.float32, shape=[None, n_classes], name="y")


# Return the keep probability to tensor
# return: Keep probability of tensor
def neural_net_keep_prob_input():
    return tf.placeholder(tf.float32, name="keep_prob")


tf.reset_default_graph()
tests.test_nn_image_inputs(neural_image_input)
tests.test_nn_label_inputs(neural_label_input)
tests.test_nn_keep_inputs(neural_net_keep_prob_input)


def new_weight(shape):
    # Generate new weight for layers
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    # Generate new biases for layers
    return tf.Variable(tf.constant(0.05, shape=[length]))


# Apply convolution then max pooling to x_tensor
#  a_tensor: TensorFlow Tensor
#  conv_num_outputs: the convolutional layer's number of outputs
#  conv_strides: Stride 2D Tuple to convolution
#  pool_ksize: the size of kernel of pool is 2D Tuple
#  pool_strides: the size of kernel of pool is 2D Tuple
#  This is the tenor represents max pooling and convolution of x_tensor
def convolution_maxpool(a_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):

    # Input's number
    conv_num_inputs = a_tensor.get_shape().as_list()[3]

    conv_height = conv_ksize[0]
    conv_width = conv_ksize[1]

    conv_shape = [conv_height, conv_width, conv_num_inputs, conv_num_outputs]

    # Weights updating before building neural networks
    weights = new_weight(shape=conv_shape)

    a_tensor = tf.nn.conv2d(input=a_tensor,
                            filter=weights,
                            strides=[1, conv_strides[0], conv_strides[1], 1],
                            padding='SAME')

    # Biases Updating before applying them
    biases = new_biases(length=conv_num_outputs)
    a_tensor += biases

    # Applying the function that is activation
    a_tensor = tf.nn.elu(a_tensor)

    a_tensor = tf.nn.max_pool(value=a_tensor,
                              ksize=[1, conv_height, conv_width, 1],
                              strides=[1, pool_strides[0], pool_strides[1], 1],
                              padding='SAME')

    return a_tensor



tests.test_con_pool(convolution_maxpool)


# Flat a_tensor to (Flattened Image Size,Batch Size)
# a_tensor: we mentioned before
# return: A tensor of Batch Size and Flattened Image Size.
def flat(a_tensor):

    # The input layer's shape is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]
    layer_shape = a_tensor.get_shape()

    # The features' number: img_height * img_width * num_channels
    # Use TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer as [num_images, num_features].
    a_tensor = tf.reshape(a_tensor, [-1, num_features])

    return a_tensor


tests.test_flat(flat)


# Using weight and bias to apply a connected layer to a_tensor
# a_tensor: A 2D tensor and its first dimension is batch size
# num_outputs: The output's number
# return: A 2D tensor.In this tensor the second dimension is num_outputs.
def fully_conn(a_tensor, num_outputs):
    num_inputs = a_tensor.get_shape().as_list()[1]
    weights = new_weight(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.add(tf.matmul(a_tensor, weights), biases)

    layer = tf.nn.elu(layer)

    return layer


tests.test_fully_conn(fully_conn)


# Using weight and bias to apply a output layer to a_tensor
# a_tensor: A 2D tensor and its first dimension is batch size
# num_outputs: The output's number
# return: A 2D tensor. In this tensor the second dimension is num_outputs.
def output(a_tensor, num_outputs):
    num_inputs = a_tensor.get_shape().as_list()[1]
    weights = new_weight(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.add(tf.matmul(a_tensor, weights), biases)

    return layer



tests.test_output(output)


# Generate a neural network model
# : a: Image data is holded by this placeholder tensor
# : keep_prob: Dropout keep probability is holded by this placeholder tensor.
# : return: The log tensor

def conv_net(a, keep_prob):

    a1 = convolution_maxpool(a_tensor=a, conv_num_outputs=32, conv_ksize=(5, 5), conv_strides=(2, 2), pool_ksize=(2, 2), pool_strides=(1, 1))
    a1 = tf.nn.dropout(a1, keep_prob)

    # Layer 2
    a2 = convolution_maxpool(a_tensor=a1, conv_num_outputs=64, conv_ksize=(5, 5), conv_strides=(1, 1), pool_ksize=(2, 2), pool_strides=(1, 1))
    a2 = tf.nn.dropout(a2, keep_prob)
    a2 = flat(a_tensor=a2)

    a2 = fully_conn(a_tensor=a2,
                    num_outputs=128)
    a2 = fully_conn(a_tensor=a2,
                    num_outputs=256)

    # Function Definition from Above:
    # output(a_tensor, num_outputs)

    a2 = output(a_tensor=a2,num_outputs=10)
    # return output
    return a2


# Reset the tensor
tf.reset_default_graph()

# Inputs
x = neural_image_input((32, 32, 3))
y = neural_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Give the logits tensor a name,
# to make is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

tests.test_conv_net(conv_net)


# Optimize the session
# session: Current TensorFlow session
# optimizer: TensorFlow optimizer function
# keep_probability: keep probability
# feature_batch: Batch of Numpy image data
# label_batch: Batch of Numpy label data
def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    session.run(optimizer, feed_dict={x: feature_batch, y: label_batch,keep_prob: keep_probability})


tests.test_train_nn(train_neural_network)


# Print out the loss and validation accuracy
# session: Current TensorFlow session
# feature_batch: Batch of Numpy image data
# label_batch: Batch of Numpy label data
# cost: TensorFlow cost function
# accuracy: TensorFlow accuracy function
def print_stats(session, feature_batch, label_batch, cost, accuracy):

    cost_score, accuracy_score = session.run([cost, accuracy], feed_dict={x: valid_features, y: valid_labels,  keep_prob: 1})
    print("cost_score = {0:>6.4}, accuracy_score = {1:>3.4%}".format(cost_score, accuracy_score))


epochs = 20
batch_size = 128
keep_probability = 0.7
print('A single batch training checking')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        batch_i = 1
        for batch_features, batch_labels in explore.load_preprocess_training_batch(batch_i, batch_size):
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)


save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in explore.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)

    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)


get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

import pickle
import explore
import random
import tensorflow as tf
# Set the size of batch
try:
    if batch_size:
        pass
except NameError:
    batch_size = 64

save_model_path = './image_classification'
n_samples = 4
top_n_predictions = 3


# Test the saved model to against the test dataset
def test_model():

    test_features, test_labels = pickle.load(open('preprocess_training.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

        # Get accuracy in batches when it meet memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0

        for train_feature_batch, train_label_batch in explore.batch_features_labels(test_features, test_labels,
                                                                                   batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total / test_batch_count))

        # Print out random samples
        random_test_features, random_test_labels = tuple(
            zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        explore.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)


test_model()
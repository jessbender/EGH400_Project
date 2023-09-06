import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras import activations
from tensorflow.keras.regularizers import l2

import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Batch size of dataset
BATCH_SIZE = 32
# Width and height of image - probably don't change this unless you change the dataset
IMAGE_SIZE = 32
# random seed to use for dataset creation.
SEED = 42
AUTO = tf.data.AUTOTUNE

# flags to indicate which models to run. You can use these if you want to run just one model.
# APPROACH_1 = True
# APPROACH_2 = True
APPROACH_3 = True

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
noosa1_02 = np.load('../EGH400_Pre_Processing/patches/Noosa1_02_patches_32x32.npz', allow_pickle=True)
noosa1_03 = np.load('../EGH400_Pre_Processing/patches/Noosa1_03_patches_32x32.npz', allow_pickle=True)
noosa1_04 = np.load('../EGH400_Pre_Processing/patches/Noosa1_04_patches_32x32.npz', allow_pickle=True)
noosa1_05 = np.load('../EGH400_Pre_Processing/patches/Noosa1_05_patches_32x32.npz', allow_pickle=True)
noosa2_01 = np.load('../EGH400_Pre_Processing/patches/Noosa2_01_patches_32x32.npz', allow_pickle=True)
noosa2_02 = np.load('../EGH400_Pre_Processing/patches/Noosa2_02_patches_32x32.npz', allow_pickle=True)

patches = []
for i in noosa1_02:
    patches.append(noosa1_02[i])
for i in noosa1_03:
    patches.append(noosa1_03[i])
for i in noosa1_04:
    patches.append(noosa1_04[i])
for i in noosa1_05:
    patches.append(noosa1_05[i])
for i in noosa2_01:
    patches.append(noosa2_01[i])
for i in noosa2_02:
    patches.append(noosa2_02[i])

p = np.stack(patches)

# np.random.shuffle(p)
split = math.ceil(p.shape[0]*0.8)
x_train, x_test = p[:split, :], p[split:, :]

# x_train = x_train / 255.0
# x_test = x_test / 255.0


def flip_random_crop(x):
    # With random crops we also apply horizontal flipping.
    x = tf.image.random_flip_left_right(x)

    rand_size = tf.random.uniform(shape=[], minval=int(0.9 * IMAGE_SIZE), maxval=1 * IMAGE_SIZE, dtype=tf.int32)
    x = tf.image.random_crop(x, (rand_size, rand_size, 1))
    x = tf.image.resize(x, (IMAGE_SIZE, IMAGE_SIZE))

    return x


def color_jitter(x, strength=[0.4, 0.4, 0.4, 0.1]):
    x = tf.image.random_brightness(x, max_delta=0.8 * strength[0])
    x = tf.image.random_contrast(
        x, lower=1 - 0.8 * strength[1], upper=1 + 0.8 * strength[1]
    )
    x = tf.image.random_saturation(
        x, lower=1 - 0.8 * strength[2], upper=1 + 0.8 * strength[2]
    )
    x = tf.image.random_hue(x, max_delta=0.2 * strength[3])

    # Affine transformations can disturb the natural range of
    # RGB images, hence this is needed.
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x


# def color_drop(x):
#     x = tf.image.rgb_to_grayscale(x)
#     x = tf.tile(x, [1, 1, 3])
#     return x


def solarize(x):
    x = tf.where(x < 0.05, x, 1.0 - x)
    return x


def blur(x):
    s = np.random.random()
    return tfa.image.gaussian_filter2d(image=x, sigma=s)


def random_apply(func, x, p):
    if tf.random.uniform([], minval=0, maxval=1) < p:
        return func(x)
    else:
        return x


def custom_augment(image):
    # As discussed in the SimCLR paper, the series of augmentation
    # transformations (except for random crops) need to be applied
    # randomly to impose translational invariance.
    image = flip_random_crop(image)
    # image = random_apply(color_jitter, image, p=0.8)
    image = random_apply(blur, image, p=0.2)
    # image = random_apply(solarize, image, p=0.2)
    # image = random_apply(color_drop, image, p=0.2)
    return image

# first dataset
ssl_ds_one = tf.data.Dataset.from_tensor_slices(x_train)
ssl_ds_one = (ssl_ds_one.shuffle(1024, seed=SEED).map(custom_augment, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO))

# second dataset - identical settings to the first
ssl_ds_two = tf.data.Dataset.from_tensor_slices(x_train)
ssl_ds_two = (ssl_ds_two.shuffle(1024, seed=SEED).map(custom_augment, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO))

# combine both the datasets, meaning that when we draw a sample we'll get image pairs, but with different augmentations
# applied to each image
ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))

# samples = next(iter(ssl_ds))
# plt.figure(figsize=(20, 10))
# for n in range(25):
#     ax = plt.subplot(5, 10, (n + 1)*2 - 1)
#     plt.imshow(samples[0][n].numpy())
#     plt.axis("off")
#
#     ax = plt.subplot(5, 10, (n + 1)*2)
#     plt.imshow(samples[1][n].numpy())
#     plt.axis("off")
# plt.show()


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(inputs, filters, num_res_blocks, pool_size):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.

    # Arguments
        inputs (layer):         the input tensor
        filters ([int]):        number of filters in each stage, length of list determines number of stages
        num_res_blocks (int):   number of residual blocks per stage
        pool_size (int):        size of the average pooling at the end

    # Returns
        output after global average pooling and flatten, ready for output
    """
    x = resnet_layer(inputs=inputs,
                     num_filters=filters[0])

    # Instantiate the stack of residual units
    for stack, filters in enumerate(filters):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=pool_size)(x)
    y = Flatten()(x)

    return y


def resnet_v2(inputs, filters, num_res_blocks, pool_size):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.

    # Arguments
        inputs (layer):         the input tensor
        filters ([int]):        number of filters in each stage, length of list determines number of stages
        num_res_blocks (int):   number of residual blocks per stage
        pool_size (int):        size of the average pooling at the end

    # Returns
        output after global average pooling and flatten, ready for output
    """

    x = resnet_layer(inputs=inputs,
                     num_filters=filters[0],
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage, filters in enumerate(filters):
        num_filters_in = filters
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=pool_size)(x)
    y = Flatten()(x)
    return y


def ssl_model(inputs, filters=[16, 32, 64], num_res_blocks=3, pooling_size=8):
    embedding = resnet_v2(inputs, filters, num_res_blocks, pooling_size)

    return keras.Model(inputs, embedding, name='ssl_model')


# Barlow Loss Class
# The Barlow loss is based on the cross-correlation matrix, subclass of keras.losses.Loss
class BarlowLoss(keras.losses.Loss):

    # init the loss with the batch size
    def __init__(self, batch_size, lambda_amt=5e-5):
        super(BarlowLoss, self).__init__()
        # lambda, used when summing the invariance term and redundancy reduction term
        self.lambda_amt = lambda_amt
        self.batch_size = batch_size

    # set the diagonals of the cross correlation matrix zero. Used in the redundancy reduction
    # loss term which takes the sum of squares of the off-diagonal values
    # inputs
    #  - A tf.tensor that represents the cross correlation matrix
    # outputs
    #  - A tf.tensor which represents the cross correlation matrix with its diagonals as zeros.
    def get_off_diag(self, c: tf.Tensor) -> tf.Tensor:
        zero_diag = tf.zeros(c.shape[-1])
        return tf.linalg.set_diag(c, zero_diag)

    # Get the barlow loss. Seek to have 1's on the diagonal, and 0's everywhere else. Loss fuction
    # is composed on two terms, the invariance term and the redundancy reduction term:
    #  - invariance term subtracts the values on the diagonal from 1, and squares the result
    #    term is minimised if the diagonal values are 1
    #  - redundancy reduction term aims to minimise the rest of the matrix (off-diagonal values), and is
    #    the sum of the square of these terms
    # Both terms are summed, with a weight, lambda, applied to the redundancy reduction term
    # inputs
    #  - A tf.tensor that represents the cross correlation matrix
    # outputs
    #  - barlow loss for the provided cross correlation matrix
    def cross_corr_matrix_loss(self, c: tf.Tensor) -> tf.Tensor:
        # subtracts diagonals from one and square the result (invariance term)
        c_diff = tf.pow(tf.linalg.diag_part(c) - 1, 2)

        # get off diagonal terms, square them, multiply by lambda (redundancy reduction term)
        off_diag = tf.pow(self.get_off_diag(c), 2) * self.lambda_amt

        # sum the terms and return the result
        loss = tf.reduce_sum(c_diff) + tf.reduce_sum(off_diag)
        return loss

    # normalise a set of predictions
    def normalize(self, output: tf.Tensor) -> tf.Tensor:
        return (output - tf.reduce_mean(output, axis=0)) / tf.math.reduce_std(
            output, axis=0)

    # create a cross-correlation matrix from two sets of predictions, each containing
    # predictions of length embedding_size. Function transposes the first of predictions,
    # and multplies these by the second to obtain a tensor of size
    # embedding_size x embedding_size. The result is divided by the batch size.
    # inputs
    #  - A normalized version of the first set of embeddings
    #  - A normalized version of the second set of embeddings
    # outputs
    #  - cross correlation matrix between the two inputs
    def cross_corr_matrix(self, z_a_norm: tf.Tensor, z_b_norm: tf.Tensor) -> tf.Tensor:
        return (tf.transpose(z_a_norm) @ z_b_norm) / self.batch_size

    # call the loss. Normalise the two tensors, create the cross correlation matrix,
    # and compute the loss
    # inputs
    #  - embeddings for the first set of augmented data
    #  - embeddings for the second set of augmented data
    # output
    #  - computed loss
    def call(self, z_a: tf.Tensor, z_b: tf.Tensor) -> tf.Tensor:
        # normalise tensors
        z_a_norm, z_b_norm = self.normalize(z_a), self.normalize(z_b)
        # compute cross correlation
        c = self.cross_corr_matrix(z_a_norm, z_b_norm)
        # compute loss
        loss = self.cross_corr_matrix_loss(c)
        return loss


# class for the Barlow Twins model
class BarlowModel(keras.Model):
    def __init__(self, encoder):
        super(BarlowModel, self).__init__()

        self.encoder = encoder
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        # get the two augmentations from the batch
        y_a, y_b = data

        # Forward pass through the encoder
        with tf.GradientTape() as tape:
            # get two versions of predictions
            z_a, z_b = self.encoder(y_a, training=True), self.encoder(y_b, training=True)
            loss = self.loss(z_a, z_b)

        # Compute gradients and update the parameters.
        grads_model = tape.gradient(loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grads_model, self.encoder.trainable_variables))

        # monitor loss
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

tf.keras.backend.clear_session()

def eval_model(ssl_model, x_train, x_test):

    # t-sne
    # compute embeddings
    embeddings = ssl_model.predict(x_test, verbose=False)
    # pass into t-sne
    tsne_embeddings = TSNE(random_state=4).fit_transform(embeddings)
    # plot the result
    fig = plt.figure(figsize=[20, 10])
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1])
    plt.show()


if (APPROACH_3):
    # create the model and compile it
    bm = BarlowModel(encoder=ssl_model(keras.Input((32, 32, 1))))
    bm.compile(optimizer=keras.optimizers.Adam(), loss=BarlowLoss(BATCH_SIZE))

    # Train this for 20 epochs again
    history = bm.fit(ssl_ds, epochs=4, verbose=True)
    plt.plot(history.history["loss"])
    plt.show()

if (APPROACH_3):
    eval_model(bm.encoder, x_train, x_test)
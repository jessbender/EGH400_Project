# Code adapted from the Barlow Twins example CAB420 author Simon Denman

import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import keras
from keras import layers
from keras.callbacks import ModelCheckpoint

from keras.layers import Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Flatten
from keras.regularizers import l2

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from matplotlib.lines import Line2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Batch size of dataset
BATCH_SIZE = 10
# Width and height of image - probably don't change this unless you change the dataset
IMAGE_SIZE = 32
# random seed to use for dataset creation.
SEED = 42
AUTO = tf.data.AUTOTUNE

lr = 1e-05
epochs = 15
lambda_val = 5e-05


# Load data
def load_data(fp):
    total_count = 0
    load_patches = []
    i = 0
    for file_path in fp:
        data = np.load(file_path, allow_pickle=True)
        for i in data:
            load_patches.append(data[i])
        count = int(i.split('_')[1])
        total_count = total_count + count
    print(f'The total number of patches is: {total_count}')

    return np.stack(load_patches)


def flip_random_crop(x):
    # With random crops we also apply horizontal flipping.
    x = tf.image.random_flip_left_right(x)

    rand_size = tf.random.uniform(shape=[], minval=int(0.9 * IMAGE_SIZE), maxval=1 * IMAGE_SIZE, dtype=tf.int32)
    x = tf.image.random_crop(x, (rand_size, rand_size, 1))
    x = tf.image.resize(x, (IMAGE_SIZE, IMAGE_SIZE))

    return x


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
    image = random_apply(blur, image, p=0.2)
    image = random_apply(solarize, image, p=0.2)
    return image


def figure_aug(image):
    im = [np.array(image), np.array(flip_random_crop(image)), np.array(blur(image)), np.array(solarize(image))]
    label = ['Original Patch', 'Flip and Random Crop', 'Blur', 'Solarize']
    plt.figure(figsize=(20, 10))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(im[i])
        plt.axis("off")
        plt.title(label[i])
    plt.savefig('Example_Augmentation.png')
    # plt.show()
    plt.close()


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
    def __init__(self, batch_size, lambda_amt=lambda_val):
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


def plot_tsne(tsne_embed, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_embed[:, 0], tsne_embed[:, 1])
    plt.title(title)
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')

    plt.savefig(f'{title}.png')
    # plt.show()
    plt.close()


# 'embeddings' containing the output embeddings from the SSL model
# 'eps' is the maximum distance between two samples for one to be considered as in the neighborhood of the other.
# 'min_samples' is the number of samples (or total weight) in a neighborhood for a point to be considered as a core
# point.
def plot_cluster_tsne(tsne_embeddings, labels):
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=labels, cmap='rainbow')
    plt.title("t-SNE Visualisation of Clusters")
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')
    plt.savefig('TSNE_Clustered')
    # plt.show()
    plt.close()


def eval_model(ssl_model, x_train, x_test):
    # t-sne
    # compute embeddings
    embeddings = ssl_model.predict(x_test, verbose=False)
    # pass into t-sne
    tsne_embeddings = TSNE(random_state=4).fit_transform(embeddings)
    # plot the result
    plot_tsne(tsne_embeddings, "TSNE Visualisation")


# Function to save model weights
def save_model(model, filepath):
    model.encoder.save(filepath)
    print(f"Model saved to {filepath}")


# Function to check if model weights exist
def model_exist(model_dir):
    if os.path.exists(model_dir):
        model_ex = True
    else:
        model_ex = False
    return model_ex


def load_model(filepath):
    loaded_model = BarlowModel(encoder=ssl_model(keras.Input((32, 32, 1))))
    loaded_model.encoder = tf.keras.models.load_model(filepath)
    print(f"Model loaded from {filepath}")
    return loaded_model


class SafeModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, checkpoint_callback):
        super(SafeModelCheckpoint, self).__init__(filepath=filepath)
        self.model_checkpoint_callback = checkpoint_callback

    def on_epoch_end(self, epoch, logs=None):
        # Check if the 'val_loss' key is in the logs and the value is not NaN
        if 'val_loss' in logs and not tf.math.is_nan(logs['val_loss']):
            self.model_checkpoint_callback.on_epoch_end(epoch, logs)


def evaluation(model):
    eval_model(model, x_train, x_test)

    embeddings = model.predict(x_test)
    # embeddings = model.predict(map_patches)

    # dbscan = DBSCAN(eps=0.5, min_samples=5)
    # labels = dbscan.fit_predict(embeddings)

    # Create and fit the GMM model
    n_components = 4  # adjust this based on data
    gmm = GaussianMixture(n_components=n_components)
    labels = gmm.fit_predict(embeddings)

    tsne = TSNE(n_components=2)
    tsne_embeddings = tsne.fit_transform(embeddings)
    plot_cluster_tsne(tsne_embeddings, labels)

    # Convert the embeddings to a NumPy array so they are comparable to the patches array
    embeddings_array = np.array(embeddings)
    labels_array = np.array(labels)

    # map_plot('2-02', labels_array)

    # Select the first 10 embeddings with the matching label
    # selected_embeddings = [embeddings_array[i] for i in indices[:10]]

    # Assuming you want to select the first 10 patches and their embeddings
    selected_patches = patches[:10]
    selected_embeddings = embeddings_array[:10]

    # Compute cosine similarities between the selected embeddings and all embeddings
    similarities = cosine_similarity(selected_embeddings, embeddings_array)

    # Set the diagonal of the similarities matrix to a low value (e.g., -1) to avoid self-similarity
    np.fill_diagonal(similarities, -1)

    # Find the indices of patches with highest similarity for each selected patch
    most_similar_indices = np.argmax(similarities, axis=1)
    selected_and_similar_patches = []

    # Print the most similar patches for each selected patch
    for i, patch in enumerate(selected_patches):
        similar_patch_index = most_similar_indices[i]
        similar_patch = patches[similar_patch_index]

        print(
            f"Selected Patch {i} (label {labels_array[i]}) is similar to Patch {similar_patch_index} (label {labels_array[similar_patch_index]})")

        # Append the selected patch and its most similar counterpart to the array
        selected_and_similar_patches.append((patch, similar_patch))

    # Convert the list to a NumPy array
    selected_and_similar_patches = np.array(selected_and_similar_patches)

    # Save the array to a file
    # np.save(f"{training_path}/selected_and_similar_patches.npy", selected_and_similar_patches)
    np.save(f"train/selected_and_similar_patches.npy", selected_and_similar_patches)

    # patch_sim = f'{training_path}/PatchSim'
    patch_sim = f'train/PatchSim'
    if not os.path.exists(patch_sim):
        os.makedirs(patch_sim)

    # Create a plot to visualize the selected patches and their most similar counterparts
    plt.figure(figsize=(12, 8))

    for i, (patch, similar_patch_index) in enumerate(zip(selected_patches, most_similar_indices)):
        similar_patch = patches[similar_patch_index]

        # Create a new figure for each pair of patches
        plt.figure()

        # Plot the selected patch
        plt.subplot(1, 2, 1)
        plt.imshow(patch.squeeze(), cmap='gray')
        plt.title(f"Selected Patch {i}")

        # Plot the most similar patch
        plt.subplot(1, 2, 2)
        plt.imshow(similar_patch.squeeze(), cmap='gray')
        plt.title(f"Similar Patch {similar_patch_index}")

        # Save the figure as an image (change the filename as needed)
        plt.savefig(f'{patch_sim}/patch_pair_{i}.png')

        # Close the current figure to release resources
        plt.close()

        # images = [f for f in os.listdir() if '.png' in f.lower()]
        #
        # for image in images:
        #     new_path = f'{training_path}/' + image
        #     shutil.move(image, new_path)


def save_parameters(training_filepath, learn_rate, batch_size, epoch, lambda_value):
    # List of hyperparameter values
    hyperparameters = [
        f"learning_rate={learn_rate}",
        f"batch_size={batch_size}",
        f"epochs={epoch}",
        f"lambda={lambda_value}"
    ]

    # File path where you want to save the hyperparameters
    file_path = f"{training_filepath}/hyperparameters.txt"

    # Open the file in write mode and write the hyperparameters
    with open(file_path, 'w') as file:
        for parameter in hyperparameters:
            file.write(parameter + "\n")

    print("Hyperparameters have been written to", file_path)


def map_plot(map_number, labels_arr):
    im = cv2.imread('../EGH400_Pre_Processing/A0/' + f'MapNoosaArea{map_number}.png', cv2.IMREAD_GRAYSCALE)
    image_height, image_width = im.shape
    chosen_size = 320
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(im)

    # Create grid lines with the specified grid cell size
    x_ticks = range(0, image_width - chosen_size, chosen_size)
    y_ticks = range(0, image_height - chosen_size, chosen_size)

    # Set the ticks based on the grid cell size
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Rotate x-axis labels for better readability
    ax.set_xticklabels(x_ticks, rotation=45)

    # Calculate the coordinates for the center of each grid cell
    x_centers = [x + chosen_size / 2 for x in x_ticks]
    y_centers = [y + chosen_size / 2 for y in y_ticks]

    # Draw a red dot in the center of each grid cell - eventually for clustering visualisation
    i = 0
    for y in y_centers:
        for x in x_centers:
            if labels_arr[i] == 0:
                ax.plot(x, y, marker='o', color='b', markersize=2) # low points
            elif labels_arr[i] == 1:
                ax.plot(x, y, marker='o', color='lawngreen', markersize=2) # white
            elif labels_arr[i] == 2:
                ax.plot(x, y, marker='o', color='r', markersize=2) # high points
            elif labels_arr[i] == 3:
                ax.plot(x, y, marker='o', color='yellow', markersize=2) # middle
            i += 1

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='n = 0', markerfacecolor='b', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='n = 1', markerfacecolor='lawngreen', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='n = 2', markerfacecolor='r', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='n = 3', markerfacecolor='yellow', markersize=10)]
    # Add grid lines
    ax.grid()
    ax.legend(handles=legend_elements, loc='upper right')
    plt.title('Depth Map Patch Extraction')
    plt.xlabel('Pixels')
    plt.ylabel('Pixels')
    plt.savefig(f'../EGH400_Pre_Processing/A0/MapNoosaArea{map_number}_w_Grid.png')


# ---------------------------------MAIN------------------------------------------------
# TODO: Import the patches from EGH400 Pre-Processing directory
# directory = '../EGH400_Pre_Processing/patches/fromA0/'
# pattern = 'from_A0_320x320.npz'

directory = '../EGH400_Pre_Processing/patches/'
pattern = 'reduced_white.npz'
file_paths = []

for filename in os.listdir(directory):
    if filename.endswith(pattern):
        file_paths.append(directory + filename)

patches = load_data(file_paths)
patches = [np.expand_dims(patch, axis=-1) for patch in patches]

p = np.stack(patches)

# # Map Testing
# map_file_path = ['../EGH400_Pre_Processing/patches/fromA0/Noosa2-02_patches_from_A0_full_map.npz']
# map_patches = load_data(map_file_path)
# map_patches = [np.expand_dims(patch, axis=-1) for patch in map_patches]
# map_patches = np.stack(map_patches)

# np.random.shuffle(p)
split = math.ceil(p.shape[0]*0.75)
x_train, x_test = p[:split, :], p[split:, :]

# Example Augmentation figure for report
figure_aug(x_train[0])

# first dataset
ssl_ds_one = tf.data.Dataset.from_tensor_slices(x_train)
ssl_ds_one = (ssl_ds_one.shuffle(1024, seed=SEED).map(custom_augment, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO))

# second dataset - identical settings to the first
ssl_ds_two = tf.data.Dataset.from_tensor_slices(x_train)
ssl_ds_two = (ssl_ds_two.shuffle(1024, seed=SEED).map(custom_augment, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO))

# combine both the datasets, meaning that when we draw a sample we'll get image pairs, but with different augmentations
# applied to each image
ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))

# Plot of a sample of patches for report
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
# plt.savefig('Sample_Patches.png')
# # plt.show()
# plt.close()

# TODO: Check what directories have already been trained, create new directory based on this
training_path = f'train/LR_{lr}/LR_{lr}_EP_{epochs}'
base_directory = f'train/LR_{lr}'

# Initialize an empty list to store existing directories
existing_directories = []

# List all items (files and directories) in the base directory
items = os.listdir(base_directory)

# Iterate through the items to identify directories
for item in items:
    item_path = os.path.join(base_directory, item)  # Get the full path of the item
    if os.path.isdir(item_path):  # Check if the item is a directory
        existing_directories.append(item)  # Add the directory name to the list

# Extract numbers from the directory names and find the highest number
highest_number = 0
for directory in existing_directories:
    try:
        # Extract the number part of the directory name
        num = int(directory.split('_')[4])
        highest_number = max(highest_number, num)
    except ValueError:
        # Handle cases where the directory name doesn't end with a number
        pass

# TODO: Load existing model if there is one, otherwise train a new model
model_dir = f'train/LR_{lr}/LR_{lr}_EP_{epochs}_11/bm_saved'
if model_exist(model_dir):
    # Model exist, load the model
    saved_model = load_model(model_dir)
    evaluation(saved_model.encoder)
else:
    # Model does not exist, train the model
    if not os.path.exists(f'{training_path}_0'):
        training_path = f'{training_path}_0'
        os.makedirs(f'{training_path}')
    else:
        training_path = f'{training_path}_{highest_number + 1}'
        os.makedirs(f'{training_path}')

    # Save hyperparameters to a .txt file
    save_parameters(training_path, lr, BATCH_SIZE, epochs, lambda_val)

    # Define the checkpoint callback
    checkpoint_filepath = 'model_checkpoints/'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        mode='min',  # Minimize validation loss
        save_weights_only=True,  # Save only model weights
        verbose=1  # Display messages about the checkpoint saving
    )
    # Create a SafeModelCheckpoint instance
    safe_model_checkpoint = SafeModelCheckpoint(checkpoint_filepath, model_checkpoint_callback)

    # Create the model and compile it
    input_shape = patches[0].shape
    bm = BarlowModel(encoder=ssl_model(keras.Input(input_shape)))

    learning_rate = lr
    print('Learning Rate: ' + str(learning_rate))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    bm.compile(optimizer=optimizer, loss=BarlowLoss(BATCH_SIZE))  # change Adam() to SGD() slower?

    ep = epochs
    print('Epochs: ' + str(ep))

    # Train the model
    history = bm.fit(ssl_ds, epochs=ep, verbose=True, callbacks=[safe_model_checkpoint])

    plt.plot(history.history["loss"])
    plt.title('Training Loss over ' + str(ep) + ' Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Adding text annotation for the final loss value
    final_loss = history.history["loss"][-1]
    plt.annotate(f'Final Loss: {final_loss:.4f}',
                 xy=(ep - 1, final_loss),
                 xytext=(ep - 1, final_loss + 0.1),
                 ha='center',
                 arrowprops=dict(arrowstyle='->'))
    plt.savefig('Loss.png')
    # plt.show()
    plt.close()

    # Save the weights after training
    save_model(bm, training_path + '/bm_saved')
    evaluation(bm.encoder)

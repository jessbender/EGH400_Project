import os

from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image
from numpy import asarray
depth_map = 'MapNoosaArea2-02.png'
patch_name = 'Noosa2_02_patches_32x32.npz'
# load the image
image = Image.open('DepthMaps/' + depth_map)
# convert image to numpy array
data = asarray(image)
print('Image shape: {}'.format(data.shape))


# Load black and white image
image = cv2.imread('DepthMaps/' + depth_map, cv2.IMREAD_GRAYSCALE)

# Define the patch size and step size
patch_size = (32, 32)
step_size = 32  # To get non-overlapping patches

patches = []
image_height, image_width = image.shape

for y in range(0, image_height - patch_size[1] + 1, step_size):
    for x in range(0, image_width - patch_size[0] + 1, step_size):
        patch = image[y:y+patch_size[1], x:x+patch_size[0]]
        # Reshape the patch to [32, 32, 1] and add it to the list
        patch = patch.reshape(patch_size[1], patch_size[0], 1)
        patches.append(patch)

# Stack all the patches into a single ndarray
patch_array = np.array(patches)


# The shape of 'patch_array' will be [Number of patches, 32, 32, 1]
#
# dpi = 400
# plot_width, plot_height = 3307, 4677
# width_inches, height_inches = plot_width / dpi, plot_height / dpi
#
# # plt.figure(figsize=(width_inches, height_inches), dpi=dpi, facecolor='w', edgecolor='k', frameon=False)
# map = data
# patches = extract_patches_2d(map, patch_size=(32, 32), max_patches=20000)
# # Create the 'patches' directory if it doesn't exist
if not os.path.exists('patches'):
    os.makedirs('patches')

filtered_patches = []
for patch in patches:
    # Calculate the percentage of white pixels in the patch
    white_pixel_count = np.sum(patch == [255])  # Assuming white is represented as [255, 255, 255]
    total_pixels = patch.shape[0] * patch.shape[1]
    white_percentage = white_pixel_count / total_pixels

    # If white percentage is less than or equal to 25%, add the patch to the filtered list
    if white_percentage <= 0.25:
        # resized_patch = patch.resize((32, 32))
        filtered_patches.append(patch)

filtered_patches = np.array(filtered_patches)
count = 0
for patch in filtered_patches:
    count = count + 1

print(count)

plt.imshow(filtered_patches[0])
plt.axis('off')
plt.tight_layout()
plt.show()

# Save the filtered patches
np.savez('patches/' + patch_name, *filtered_patches)

# Show original image
# plt.imshow(map)
# plt.axis('off')
# plt.tight_layout()
# plt.show()

# load_patches = np.load('patches/Noosa2_01_patches_32x32.npz')

# Show patches in a 10x10 grid
# gridx, gridy = 10, 10
# fig, ax = plt.subplots(gridx, gridy, figsize=patch_size, dpi=400, facecolor='w', edgecolor='k',
#                        frameon=False)
#
# for i in range(gridx):
#     for j in range(gridy):
#         im = load_patches[('arr_%d' % (10 * i + j,))]
#         # print(np.amin(im), np.amax(im))
#         ax[i, j].axis('off')
#         ax[i, j].imshow(im)
#
# # Show grid
# plt.show()

# Resize to 32x32 for Barlow Twins -> Tensorflow map/resize functions
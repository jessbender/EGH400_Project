import os

from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from numpy import asarray
# load the image
image = Image.open('DepthMaps/MapNoosaArea2-02.png')
# convert image to numpy array
data = asarray(image)
print('Image shape: {}'.format(data.shape))

dpi = 400
plot_width, plot_height = 3307, 4677
width_inches, height_inches = plot_width / dpi, plot_height / dpi

# plt.figure(figsize=(width_inches, height_inches), dpi=dpi, facecolor='w', edgecolor='k', frameon=False)
map = data
patches = extract_patches_2d(map, patch_size=(100, 100), max_patches=20000)
# Create the 'patches' directory if it doesn't exist
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
        filtered_patches.append(patch)

filtered_patches = np.array(filtered_patches)
count = 0
for patch in filtered_patches:
    count = count + 1

print(count)
# Save the filtered patches
np.savez('patches/Noosa2_02_patches.npz', *filtered_patches)

# Show original image
# plt.imshow(map)
# plt.axis('off')
# plt.tight_layout()
# plt.show()

load_patches = np.load('patches/Noosa2_02_patches.npz')

# Show patches in a 10x10 grid
gridx, gridy = 10, 10
fig, ax = plt.subplots(gridx, gridy, figsize=(width_inches, height_inches), dpi=dpi, facecolor='w', edgecolor='k',
                       frameon=False)

for i in range(gridx):
    for j in range(gridy):
        im = load_patches[('arr_%d' % (10 * i + j,))]
        # print(np.amin(im), np.amax(im))
        ax[i, j].axis('off')
        ax[i, j].imshow(im)

# Show grid
plt.show()

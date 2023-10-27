import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_depth_map(map_num):
    depth_map = f'MapNoosaArea{map_num}.png'
    return depth_map


def patch_extraction(im, chosen_size):
    patches = []
    image_height, image_width = im.shape
    print(f'image h: {image_height}, w: {image_width}')

    # Define the patch size and step size
    patch_size = (chosen_size, chosen_size)
    step_size = chosen_size

    index = 0
    patch_index = []
    for y in range(0, image_height - patch_size[1] + 1, step_size):
        for x in range(0, image_width - patch_size[0] + 1, step_size):
            index += 1
            patch = im[y:y+patch_size[1], x:x+patch_size[0]]
            # Reshape the patch to [32, 32, 1] and add it to the list
            patch = patch.reshape(patch_size[1], patch_size[0], 1)
            patches.append(patch)
            patch_index.append((index, map_number))

    # Create the 'patches' directory if it doesn't exist
    if not os.path.exists('patches'):
        os.makedirs('patches')

    filtered_patches = []
    filt_index = 0
    for patch in patches:

        # Calculate the percentage of white pixels in the patch
        white_pixel_count = np.sum(patch == [255])  # Assuming white is represented as [255, 255, 255]
        total_pixels = patch.shape[0] * patch.shape[1]
        white_percentage = white_pixel_count / total_pixels

        # If white percentage is less than or equal to 10%, add the patch to the filtered list
        if white_percentage <= 0.10:
            filt_index += 1
            filtered_patches.append(patch)
        else:
            patch_index[filt_index] = (0, map_number)
            filt_index += 1

    count = 0
    for i in patch_index:
        if not i == (0, map_number):
            count += 1

    print(f'patch index: {len(patch_index)}, patch values: {count}')
    # filtered_patches = np.array(filtered_patches)
    count = 0
    resized_patches = []
    scale_down = 32/chosen_size

    for patch in filtered_patches:
        count = count + 1
        resize = cv2.resize(patch, None, fx=scale_down, fy=scale_down, interpolation=cv2.INTER_CUBIC)

        resized_patches.append(resize)

    print(f'No of patches: {count}')

    # create figure
    fig = plt.figure(figsize=(10, 7))

    # setting values to rows and column variables
    rows = 1
    columns = 2

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)
    plt.imshow(filtered_patches[0])
    plt.title(f'{chosen_size}x{chosen_size}')
    plt.axis('off')
    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
    plt.imshow(resized_patches[0])
    plt.title('32x32')
    plt.axis('off')
    plt.show()

    # Save the filtered patches
    patch_file = 'patches/' + patch_name
    np.savez(patch_file, *resized_patches)

    # Show original image
    # Show the original image
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

    # Add grid lines
    ax.grid()
    plt.title('Depth Map Patch Extraction')
    plt.xlabel('Pixels')
    plt.ylabel('Pixels')
    plt.savefig(f'DepthMaps/MapNoosaArea{map_number}_w_Grid.png')
    np.save(f'DepthMaps/MapNoosaArea{map_number}_Patch_Index', patch_index)
    # plt.show()

    # Show a sample of patches
    # load_patches = np.load(patch_file)
    #
    # # Show patches in a 10x10 grid
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
    # plt.savefig(f'patches/fromA0/MapNoosaArea{map_number}_sample.png')
    # plt.show()


map_number = ['1-02', '1-03', '1-04', '1-05', '2-01', '2-02']
map_number = map_number[0]  # Select a map to extract patches from
# Load black and white image
image = cv2.imread('DepthMaps/' + get_depth_map(map_number), cv2.IMREAD_GRAYSCALE)
patch_name = f'Noosa{map_number}_patches.npz'

# Extract patches from the input map at the chosen size
patch_extraction(image, chosen_size=150)

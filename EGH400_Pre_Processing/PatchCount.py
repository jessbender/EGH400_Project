import os

from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from numpy import asarray

noosa1_2 = np.load('patches/Noosa1_02_patches.npz')
noosa1_3 = np.load('patches/Noosa1_03_patches.npz')
noosa1_4 = np.load('patches/Noosa1_04_patches.npz')
noosa1_5 = np.load('patches/Noosa1_05_patches.npz')
noosa2_1 = np.load('patches/Noosa2_01_patches.npz')
noosa2_2 = np.load('patches/Noosa2_02_patches.npz')
count = 0
for patch in noosa1_2:
    count = count + 1

for patch in noosa1_3:
    count = count + 1

for patch in noosa1_4:
    count = count + 1

for patch in noosa1_5:
    count = count + 1

for patch in noosa2_1:
    count = count + 1

for patch in noosa2_2:
    count = count + 1
print(count)





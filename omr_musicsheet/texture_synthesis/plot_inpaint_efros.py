"""
=========================
Textural Image Inpainting
=========================

Image Inpainting is the process of reconstructing lost or deteriorated parts
of an image.

In this example we wll show Textural inpainting. Textures have repetitive
patterns and hence cannot be restored by continuing neighbouring geometric
properties into the unknown region. The correct match is found using the
minimum Sum of Squared Differences (SSD) between a patch about the pixel to
be inpainted and all other patches in the image which do not contain any
boundary region and no unknown or masked region. This implementation
updates 1 pixel at a time.


Un peu technique l'install.
Ne passe pas avec un interpreteur python >= 3.7
car la version spécifique scikit-image
du dépot: https://github.com/chintak/scikit-image
nécessite une version (ancienne) de cython==0.17 qui ne semble pas compatible
avec python 3.7.
Faudrait tenter un autre interpréteur, on cherchait une version compatible
de Cython qui peut etre avec py3.7 et faire tourner la lib scikit-image ...
"""

import numpy as np
# import matplotlib.pyplot as plt
# from skimage import datasets
from skimage.filter.inpaint_texture import inpaint_efros
import cv2

filename = "../Page_09_Pattern_24_rot.png"
image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
# image = datasets.camera()[300:500, 350:550]

mask = np.zeros_like(image, dtype=np.uint8)

# paint_region = (slice(125, 145), slice(20, 50))

# image[paint_region] = 0
# mask[paint_region] = 1
image_copy = image.copy()
image[image_copy == 255] = 0
mask[image_copy == 255] = 255

painted = inpaint_efros(image, mask, window=7)

# fig, (ax0, ax1) = plt.subplots(ncols=2)
# ax0.set_title('Input image')
# ax0.imshow(image, cmap=plt.cm.gray)
# ax1.set_title('Inpainted image')
# ax1.imshow(painted, cmap=plt.cm.gray)
# plt.show()

cv2.imshow('image', image)
cv2.imshow('mask', mask)
cv2.imshow('painted', painted)

cv2.waitKey(0)

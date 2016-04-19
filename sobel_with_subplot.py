import cv2
# import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Page_09_Pattern_26.png', 0)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
scharrx = cv2.medianBlur(cv2.Scharr(img, cv2.CV_8U, 0, 1), 5)
scharry = cv2.medianBlur(cv2.Scharr(img, cv2.CV_8U, 0, 1), 5)

plt.subplot(3, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 2, 2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 2, 3), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 2, 4), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 2, 5), plt.imshow(scharrx, cmap='gray')
plt.title('Scharr X'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 2, 6), plt.imshow(scharry, cmap='gray')
plt.title('Scharr Y'), plt.xticks([]), plt.yticks([])

plt.show()

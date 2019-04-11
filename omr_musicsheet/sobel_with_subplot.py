# import numpy as np
from matplotlib import pyplot as plt
from omr_musicsheet.cv2_tools import *

img = cv2.imread('Page_09_Pattern_26.png', 0)
img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
size_kernel = 3
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=size_kernel)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=size_kernel)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0.0)
sobelx = cv2.Sobel(sobelxy, cv2.CV_64F, 1, 0, ksize=size_kernel)
sobely = cv2.Sobel(sobelxy, cv2.CV_64F, 0, 1, ksize=size_kernel)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0.0)
# sobelxy = cv2.convertScaleAbs(sobelxy)

scharrx = cv2.medianBlur(cv2.Scharr(img, cv2.CV_8U, 0, 1), 3)
scharry = cv2.medianBlur(cv2.Scharr(img, cv2.CV_8U, 0, 1), 3)
scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0.0)
scharrxy = cv2.convertScaleAbs(scharrxy)

plt.subplot(3, 3, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 3), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 4), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 5), plt.imshow(scharrx, cmap='gray')
plt.title('Scharr X'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 6), plt.imshow(scharry, cmap='gray')
plt.title('Scharr Y'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 7), plt.imshow(sobelxy, cmap='gray')
plt.title('sobelxy'), plt.xticks([]), plt.yticks([])

plt.show()

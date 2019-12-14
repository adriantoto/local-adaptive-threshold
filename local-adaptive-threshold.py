# Import Packages
import cv2
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
import numpy as np

# Gridspec (Untuk mengatur letak gambar output)
gs = gridspec.GridSpec(5, 6)

# File path (sesuai directory file input)
path = r'D:\User Projects\PCD\tugasPCD\image-sample.JPG'
# import file
original = cv2.imread(path)
""" --- Show the Image --- """
plt.subplot(gs[0:2, 0:2])
plt.title('1. Original Image')
plt.imshow(original, cmap="gray", vmin=0, vmax=255)

"""Preprocessing"""

# Convert image in grayscale
gray_im = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
""" --- Show the Image --- """
plt.subplot(gs[0:2, 2:4])
plt.title('2. Grayscale Image')
plt.imshow(gray_im, cmap="gray", vmin=0, vmax=255)

# Contrast adjusting with gamma correction y = 1.2
gray_correct = np.array(255 * (gray_im / 255) ** 1.2 , dtype='uint8')
""" --- Show the Image --- """
plt.subplot(gs[0:2, 4:6])
plt.title('3. Contrast Adjusting')
plt.imshow(gray_correct, cmap="gray", vmin=0, vmax=255)

# Contrast adjusting with histogramm equalization
gray_equ = cv2.equalizeHist(gray_im)
""" --- Show the Image --- """
#plt.subplot(222)
#plt.title('Histogram equilization')
#plt.imshow(gray_correct, cmap="gray", vmin=0, vmax=255)

""" ---Processing---  """
# Local adaptive threshold
thresh = cv2.adaptiveThreshold(gray_correct, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 19)
thresh = cv2.bitwise_not(thresh)
""" --- Show the Image --- """
plt.subplot(gs[3:5, 0:2])
plt.title('4. Local Adaptive Threshold')
plt.imshow(thresh, cmap="gray", vmin=0, vmax=255)

# Dilation & erosion
kernel = np.ones((15,15), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
img_erode = cv2.erode(img_dilation,kernel, iterations=1)
# clean all noise after dilatation and erosion
img_erode = cv2.medianBlur(img_erode, 7)
""" --- Show the Image --- """
plt.subplot(gs[3:5, 2:4])
plt.title('5. Dilation & erosion')
plt.imshow(img_erode, cmap="gray", vmin=0, vmax=255)

# Labeling
ret, labels = cv2.connectedComponents(img_erode)
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[label_hue == 0] = 0
""" --- Show the Image --- """
plt.subplot(gs[3:5, 4:6])
plt.title('6. Objects counted:'+ str(ret-1))
plt.imshow(labeled_img)

# Output
print('objects number is:', ret-1)
plt.show()
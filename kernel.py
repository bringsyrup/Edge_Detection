"""
This module takes an image and outputs the grey-scaled image convolved with a given kernel.
For now, the kernel is hard-coded for edge-detection
"""
import numpy as np
from PIL import Image

fileName = 'checkers.jpg'
img = Image.open(fileName)
img = img.convert('L')


img.save('black_and_white_{0}'.format(fileName))

imgArray = np.array(img, dtype = 'uint8')

kernel = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
# kernel = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])

def convolve(imgArray, kernel):
    """colvolve the image matrix with the kernel"""
    shape = imgArray.shape
    convolvedArray = np.zeros(shape)
    kernalIterator = [-1, 0, 1]
    for x in range(1, shape[0] - 1):
        for y in range(1, shape[1] -1):
            for i in kernalIterator:
                for j in kernalIterator:
                    convolvedArray[x, y] +=  (imgArray[x + i, y + j]*
                            kernel[1 + i, 1 + j])
    return convolvedArray


convolvedArray = convolve(imgArray, kernel)
avg = (np.amax(convolvedArray) + np.amin(convolvedArray))/2

for i in xrange(convolvedArray.shape[0]):
  for j in xrange(convolvedArray.shape[1]): 
    if convolvedArray[i][j] < avg:
      convolvedArray[i][j] = 0
    elif convolvedArray[i][j] >= avg:
      convolvedArray[i][j] = 255

newImg = Image.fromarray(convolvedArray.astype('uint8'), "L")

newImg.save('convolved_{0}'.format(fileName))


import numpy as np
from PIL import Image


fileName = 'lena.jpg'
img = Image.open(fileName)
img = img.convert('L')

img.save('black_and_white_{0}'.format(fileName))

imgArray = np.array(img, dtype = float)
kernel = np.array([[-1, -1, -1][-1, 8, -1][-1, -1, -1]])

def convolve(imgArray, kernel):
  shape = imgArray.shape
  new = np.zeros(shape)
  for x in range(1, shape[0] - 1):
    # for y in range
    pass

print imgArray
print type(imgArray)


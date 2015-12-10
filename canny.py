import numpy as np
from PIL import Image

class EdgeDetection():
    def __init__(self, fileName):
        self.kernel = np.array([[[1, 0, -1],[2, 0, -2],[1, 0, -1]], [[1, 2, 1],[0, 0, 0],[-1, -2, -1]]])
        self.smooth = np.array([[1./16, 1./8, 1./16],[1./8, 1./4, 1./8],[1./16, 1./8, 1./16]])
        self.fileName = fileName
        self.getImg()
        self.convolve()
        self.getDirection()
        self.suppress()
        array = self.gradientArray
        self.threshold(.1, .2)
        self.link()
        array = self.upperArray

        # array = self.binarize(array)
        newImg = Image.fromarray(array.astype('uint8'), "L")
        newImg.save("canny_{0}".format(self.fileName))
        # newImg2 = Image.fromarray(self.upperArray.astype('uint8'), "L")
        # newImg2.save("canny_upperThresh_{0}".format(self.fileName))        
        # newImg3 = Image.fromarray(self.lowerArray.astype('uint8'), "L")
        # newImg3.save("canny_lowerThresh_{0}".format(self.fileName))

    def getImg(self):
        self.img = Image.open(self.fileName)
        self.img = self.img.convert('L')
        self.imgArray = np.array(self.img, dtype = 'uint8')
        self.shape = self.imgArray.shape

    def _convolve(self, kernel):
        """helper function to convolve images with 1d kernel"""
        print "convolving"
        convolvedArray = np.zeros(self.shape)
        kernelIterator = [-1, 0, 1]
        for x in range(1, self.shape[0] - 1):
            for y in range(1, self.shape[1] -1):
                for i in kernelIterator:
                    for j in kernelIterator:
                        convolvedArray[x, y] +=  (self.imgArray[x + i, y + j]*
                                kernel[1 + i, 1 + j])
        return convolvedArray

    def convolve(self):
        """colvolve the image matrix with the kernel"""
        self.imgArray = self._convolve(self.smooth)

        convolvedArrayX = self._convolve(self.kernel[0])
        convolvedArrayY = self._convolve(self.kernel[1])
        self.gradientArray = np.sqrt(convolvedArrayX**2 + convolvedArrayY**2)
        self.directionArray = np.arctan2(convolvedArrayY, convolvedArrayX)*180/np.pi

    def getDirection(self):
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if (self.directionArray[x][y]<22.5 and self.directionArray[x][y]>=0) or \
                   (self.directionArray[x][y]>=157.5 and self.directionArray[x][y]<202.5) or \
                   (self.directionArray[x][y]>=337.5 and self.directionArray[x][y]<=360):
                    self.directionArray[x][y]=0
                elif (self.directionArray[x][y]>=22.5 and self.directionArray[x][y]<67.5) or \
                     (self.directionArray[x][y]>=202.5 and self.directionArray[x][y]<247.5):
                    self.directionArray[x][y]=45
                elif (self.directionArray[x][y]>=67.5 and self.directionArray[x][y]<112.5)or \
                     (self.directionArray[x][y]>=247.5 and self.directionArray[x][y]<292.5):
                    self.directionArray[x][y]=90
                else:
                    self.directionArray[x][y]=135

    def suppress(self):
        for x in range(1, self.shape[0] - 1):
            for y in range(1, self.shape[1] - 1):
                if self.directionArray[x][y] == 0:
                    neighbor1 = self.gradientArray[x][y + 1]
                    neighbor2 = self.gradientArray[x][y - 1]
                elif self.directionArray[x][y] == 45:
                    neighbor1 = self.gradientArray[x + 1][y - 1]
                    neighbor2 = self.gradientArray[x - 1][y + 1]
                elif self.directionArray[x][y] == 90:
                    neighbor1 = self.gradientArray[x + 1][y]
                    neighbor2 = self.gradientArray[x - 1][y]
                elif self.directionArray[x][y] == 135:
                    neighbor1 = self.gradientArray[x + 1][y + 1]
                    neighbor2 = self.gradientArray[x - 1][y - 1]

                if self.gradientArray[x][y] <= max(neighbor1, neighbor2):
                    self.gradientArray[x][y] = 0

    def threshold(self, lower, upper):
        """
            upper: percentage for upper threshold
            lower: percentage for lower threshold
        """
        span = (np.amax(self.gradientArray) + np.amin(self.gradientArray))
        upper = upper*span
        lower = lower*span

        self.upperArray = np.zeros(self.shape)
        self.lowerArray = np.zeros(self.shape)

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if self.gradientArray[x][y] >= upper:
                    self.upperArray[x][y] = 255
                if self.gradientArray[x][y] < upper:
                    if  self.gradientArray[x][y] >= lower:
                        self.lowerArray[x][y] = 255

    def link(self):
        def traverseEdges(x, y):
            adjacentInd = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            for i, j in adjacentInd:
                if not self.upperArray[x + i][y + j]:
                    if self.lowerArray[x + i][y + j]:
                        self.upperArray[x + i][y + j] = 255
                        traverseEdges(x + i, y + j)

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if self.upperArray[x][y]:
                    traverseEdges(x, y)


    def binarize(self, array):
        """convert array into black and white image"""
        print "binarizing"
        avg = (np.amax(array) + np.amin(array))/2
        for i in xrange(array.shape[0]):
            for j in xrange(array.shape[1]): 
                if array[i][j] < avg:
                    array[i][j] = 0
                elif array[i][j] >= avg:
                    array[i][j] = 255
        return array


def main():
    import argparse as ap
    parser = ap.ArgumentParser(
            description = "This module filters images according to the options below",
            formatter_class = ap.RawTextHelpFormatter
            )
    parser.add_argument("image",
            type = str,
            help = "Choose an image file for processing. Currently tested filetypes include: .bmp, .png, .jpg")
    
    args = parser.parse_args()




    try:
        print "filtering in process"
        EdgeDetection(args.image)
        
    except IOError:
        print "please make sure the image file you are trying to filter exists"

    return

if __name__ == '__main__':
    main()
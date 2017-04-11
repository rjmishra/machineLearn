from PIL import Image
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt


imOne = Image.open('band1.gif')
imTwo = Image.open('band2.gif')
imThree = Image.open('band3.gif')
imFour = Image.open('band4.gif')

arrOne = np.asarray(imOne)
arrTwo = np.asarray(imTwo)
arrThree = np.asarray(imThree)
arrFour = np.asarray(imFour)
data = np.dstack([imOne, imTwo, imThree, imFour])

#river dimensions
values = []
file = open('river.txt', "r")
for line in file:
    values.append([int(n) for n in line.split(',')])
    

river_sample = []
for j in range(len(values)):
    d = [x for x in data[values[j][1]][values[j][0]]]
    d.append(0)
    
    river_sample.append(d)

river_sample = np.array(river_sample)


#land dimensions
values = []
file = open('land.txt', "r")
for line in file:
    values.append([int(n) for n in line.split(',')])

land_sample = []
for j in range(len(values)):
    d = [x for x in data[values[j][1]][values[j][0]]]
    d.append(1)
    land_sample.append(d)

land_sample = np.array(land_sample)


def distance(a, b):
    """ This function will return euclidean disatance between two points"""
    return math.sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1] - b[1])*(a[1]-b[1]) + (a[2] - b[2])*(a[2]-b[2]) + (a[3] - b[3])*(a[3]-b[3]))



img = np.zeros((512, 512))
k = 1
for m in range(len(data)):
    for n in range(len(data)):
        neighbours = [[1000000.0, 0]]
        neighbours = neighbours * k
        neighbours[0] = [100000.0, 0]
        dist = 0.0
        for i in range(len(land_sample)):
            dist = distance(data[m][n], land_sample[i])
            d = [dist, 1]
            l = 0
            while dist > neighbours[l][0] and l<k-1:
                l = l+1
            while (l < k-1):
                x = neighbours[l]
                neighbours[l] = d
                d = x
                l = l+1
            if(l < k and neighbours[l][0] > d[0] ):
                neighbours[l] = d
        for i in range(len(river_sample)):
            dist = distance(data[m][n], river_sample[i])
            d = [dist, 0]
            l = 0
            while dist > neighbours[l][0] and l<k-1:
                l = l+1
            while (l < k-1):
                x = neighbours[l]
                neighbours[l] = d
                d = x
                l = l+1
            if(l < k and neighbours[l][0] > d[0] ):
                neighbours[l] = d
        ones = 0
        zeroes = 0
        for i in range(k):
            ones = ones + neighbours[i][1]
            zeroes = k - ones
        if(ones > zeroes):
            img[m][n] = 1
        #print('Completion percentage:', ((m*512 + (n+1))*100.0)/(512*512))
    print('Completion percentage:', ((m+1)*100.0/512))





class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)


fig, ax = plt.subplots()
fig.suptitle('k-NN with k=1')
im = ax.imshow(img, interpolation='none', cmap='gray')
ax.format_coord = Formatter(im)
plt.show()


# coding: utf-8

# In[1]:

from PIL import Image


# In[2]:

imOne = Image.open('band1.gif')
imTwo = Image.open('band2.gif')
imThree = Image.open('band3.gif')
imFour = Image.open('band4.gif')


# In[3]:

import numpy as np


# In[4]:

arrOne = np.asarray(imOne)
arrTwo = np.asarray(imTwo)
arrThree = np.asarray(imThree)
arrFour = np.asarray(imFour)


# In[5]:

data = np.dstack([imOne, imTwo, imThree, imFour])


# In[6]:

data.shape


# In[7]:

#river dimensions
values = []
file = open('river.txt', "r")
for line in file:
    values.append([int(n) for n in line.split(',')])


# In[8]:

river_sample = []
for j in range(len(values)):
    river_sample.append(data[values[j][0]][values[j][1]])


# In[9]:

river_sample = np.array(river_sample)


# In[10]:

river_sample.shape


# In[11]:

mean = np.mean(river_sample, axis=0)


# In[12]:

mean


# In[13]:

riverMean = river_sample - mean


# In[14]:

covRiver = np.dot(riverMean.T, riverMean) / river_sample.shape[0]


# In[17]:

covRiver


# In[16]:

for i in range(4):
    for j in range(4):
        if i != j:
           covRiver[i][j] = 0 


# In[18]:

import scipy as scp


# In[19]:

from scipy.stats import multivariate_normal
pdfNormal = multivariate_normal(mean=mean, cov=covRiver)


# In[20]:

#land dimensions
values = []
file = open('land.txt', "r")
for line in file:
    values.append([int(n) for n in line.split(',')])


# In[21]:

land_sample = []
for j in range(len(values)):
    land_sample.append(data[values[j][0]][values[j][1]])


# In[22]:

land_sample = np.array(land_sample)


# In[24]:

land_mean = np.mean(land_sample, axis=0)
landMeanSubtract = land_sample - land_mean
covLand = np.dot(landMeanSubtract.T, landMeanSubtract) / land_sample.shape[0]

for i in range(4):
    for j in range(4):
        if i != j:
           covLand[i][j] = 0 


# In[25]:

covLand


# In[26]:

pdfNormalLand = multivariate_normal(mean=land_mean, cov=covLand)


# In[27]:

# input the prior probability of each class
priorLand = input()
priorRiver = input()


# In[28]:

priorLand = float(priorLand)
priorRiver= float(priorRiver)


# In[29]:

#if(priorLand*pdfNormalLand(data[19][29]) >= priorRiver*pdfNormal())
pdfNormalLand.pdf(data[19][29])


# In[30]:

pdfNormal.pdf(data[19][29])


# In[69]:

priorLand = float(priorLand)
priorRiver = float(priorRiver)


# In[31]:

kolkata = np.ndarray(shape=(2,2), dtype=float, order='F')


# In[32]:

priorRiver*pdfNormal.pdf(data[19][29])


# In[33]:

kolkata = np.ndarray(shape=(512,512), dtype=float, order='F')
for i in range(512):
    for j in range(512):
        if(priorLand*pdfNormalLand.pdf(data[i][j]) > priorRiver*pdfNormal.pdf(data[i][j])):
            kolkata[i][j] = 1
        else:
            kolkata[i][j] = 0


# In[76]:

#Image.Image.show(kolkata)


# In[34]:

from matplotlib import pyplot as plt


# In[35]:

#get_ipython().magic('matplotlib inline')


# In[37]:

class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)

#data = np.random.random((10,10))

fig, ax = plt.subplots()
im = ax.imshow(kolkata, interpolation='none', cmap='gray')
ax.format_coord = Formatter(im)
plt.show()


# In[ ]:




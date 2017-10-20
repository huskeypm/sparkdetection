import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy.misc import toimage
from scipy.ndimage.filters import *
from copy import copy
import numpy as np
import matplotlib.colors as colors
import matplotlib.mlab as mlab


def myplot(img,fileName=None):
    plt.axis('equal')
    plt.pcolormesh(img, cmap='gray')
    if fileName!=None:
        plt.gcf().savefig(fileName,dpi=300)
    #plt.axis('off')
    plt.tight_layout()




def ReadImg(fileName):
    img = cv2.imread(fileName)
    if img is None:
        raise RuntimeError(fileName+" likely doesn't exist")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def rotater(img, ang):
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),ang,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst


# In[5]:

def GenThreshed(image,thresh=122):
    # record values above threshold 
    superThresh = np.copy(image)
    superThresh[image <= thresh]  = 0
    plt.figure()
    pcolormesh(superThresh)

    # sort by intensity 
    flat = np.ndarray.flatten(superThresh)
    sidx = np.argsort( flat )[::-1]
    plt.figure()
    plt.plot( flat[sidx])

    # plot up to first zero 
    firstZero = np.argwhere( flat[sidx]<1e-9)[0]
    plt.xlim([0,firstZero])

    return superThresh, sidx


# In[6]:

def GetPeriod(hstrip):    
    # thresh
    hstrip[np.where(hstrip - np.mean( hstrip ) < 0 )] = 0

    sumAxis = np.argmin( np.shape(hstrip)) # find smallest axis
    
    line = np.sum( hstrip, sumAxis)

    # identify peaks (leading edge) 
    thresh = np.array( line > np.mean(line),dtype=float)
    nPix = np.shape(thresh)[0]
    diff = thresh[1:]-thresh[0:(nPix-1)]
    leadingEdges = np.where(diff>0)[0] 
    totDist = np.array(leadingEdges[-1] - leadingEdges[0],dtype=float)
    nPeaks = np.shape( leadingEdges )[0]
    peakDist = np.int(totDist/(nPeaks-1))

    return peakDist


# In[7]:

def Thresher(gray): # threshes to allow for period finding
  ymax,xmax = np.where(gray == np.max(gray))[0][0], np.where(gray == np.max(gray))[1][0]
  threshed = np.copy(gray)
  threshed[gray <= threshold]  = 0
  #print threshed
  xstrip = np.asarray( threshed[(ymax-1):(ymax+1),:], dtype=float)
  xperiod = GetPeriod(xstrip)
    
  ystrip = np.asarray( threshed[:,(xmax-1):(xmax+1)], dtype=float)
  yperiod = GetPeriod(ystrip)
  return yperiod,xperiod


# In[8]:

def Claher(img,bounds = [51,51]):
    clahe9 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(bounds[0],bounds[1]))
    cl9 = clahe9.apply(img)
    return cl9


# In[9]:

def Anchor(smoothed,yperiod,xperiod):
  leftBottom = smoothed[0:(yperiod/2),0:(xperiod/2)]
  yMin,xMin = np.unravel_index(leftBottom.argmax(), leftBottom.shape)
  xMin-=1
  return yMin,xMin


def Doing_stuff(gray, threshold = 230):
  ymax,xmax = np.where(gray == np.max(gray))[0][0], np.where(gray == np.max(gray))[1][0]
  threshed = np.copy(gray)
  threshed[gray <= threshold]  = 0
  #print threshed
  xstrip = np.asarray( threshed[(ymax-1):(ymax+1),:], dtype=float)
  xperiod = GetPeriod(xstrip)
    
  ystrip = np.asarray( threshed[:,(xmax-1):(xmax+1)], dtype=float)
  yperiod = GetPeriod(ystrip)
  return yperiod,xperiod

def doItBulk( fileName = 'clahe_Best.png', rotation = 5, region =[280,380,60,160]  ):
    threshold = 230
    Box = ReadImg('clahe_Best.jpg')
    MyBox = rotater(Box,rotation)[region[0]:region[1],region[2]:region[3]]
    y,x = np.where(MyBox == np.max(MyBox))[0][0], np.where(MyBox == np.max(MyBox))[1][0]
    gray = np.copy(MyBox)
    kernel = np.ones((3,3),np.float32)/9
    smoothed = cv2.filter2D(gray,-1,kernel)
    yperiod,xperiod = Doing_stuff(MyBox, threshold)
    yMin,xMin = Anchor(smoothed,yperiod,xperiod)
    
    vals=np.floor(gray.shape/np.array([yperiod,xperiod]))-1
    yIter=np.int(vals[0])
    xIter=np.int(vals[1])
    nImg = xIter * yIter;
    allCells = []
    for yi in np.arange(yIter):
        for xi in np.arange(xIter):
          yS=yMin + yi*yperiod+16; yF = yS+yperiod  +2       
          xS=xMin + xi*xperiod+1; xF = xS+xperiod
          unitCell= gray[yS:yF,xS:xF]
          if unitCell.any()==True:  
            allCells.append(unitCell) 
            #plt.figure()  
            #myplot(unitCell)
    stacked = np.floor(np.average(allCells,axis=0))

    bulkFilter = np.zeros_like(stacked)
    bulkFilter[stacked>=220] = 255


    
    noiseBulk = []
    for yi in np.arange(yIter):
        for xi in np.arange(xIter):
          yS=yMin + yi*yperiod+16; yF = yS+yperiod  +2         
          xS=xMin + xi*xperiod+1; xF = xS+xperiod
          unitCell= gray[yS:yF,xS:xF]

          if unitCell.any()==True:  
            allCells.append(unitCell) 
            noiseBulk.append(np.std(unitCell -stacked))
            #print noiseBulk
            #plt.figure()  
            #myplot(unitCell-stacked)
    
       
    
    return bulkFilter, noiseBulk


# In[11]:

def doItFused( fileName = 'clahe_Best.png', rotation = 30, region =[290,395,340,365]):

    Box = ReadImg('clahe_Best.jpg')
    MyBox = rotater(Box,rotation)[region[0]:region[1],region[2]:region[3]]

    
    iters = np.linspace(0,4,5, dtype = int)
    iters2 = np.linspace(0,4,5, dtype = int)
    mySum = []
    Sums = []
    #threshold = 240
    for j, why in enumerate(iters2):
        tempBox =MyBox[(why):100+(why), :]
        for i, iteration in enumerate(iters):
            unitCell = tempBox[20*iteration:(20*(1+iteration)), :]
            thisHit = np.array(unitCell, dtype = float)
            #plt.figure()
            #myplot(thisHit)
            mySum.append(unitCell) 
        #print 'printing unitCell', unitCell
        Sum2 = np.sum(mySum, axis = 0)/75
        threshed = np.copy(Sum2)
        Sums.append(threshed)
    Sum = np.sum(Sums,axis =0)

    fusedFilter = np.copy(Sum)
    #fusedFilter[stacked>=220] = 255


    
    noiseFused = []
    for j, why in enumerate(iters2):
        tempBox =MyBox[(why):100+(why), :]
        for i, iteration in enumerate(iters):
            unitCell = tempBox[20*iteration:(20*(1+iteration)), :]
            thisHit = np.array(fusedFilter,dtype = float)
            #print 'unit, sum', np.shape(unitCell), np.shape(thisHit)
            noiseFused.append(np.std(unitCell -thisHit))
            #print noiseFused

       
    
    return fusedFilter,noiseFused




import sys
sys.path.append("../mach")
import matchedFilter as mF

import bankDetect as bD 

# In[13]:

def  doCorrelation(myImg,myFilter):
    Correlated = mF.matchedFilter(myImg, myFilter)
    return Correlated
    


# In[13]:




# In[13]:




# In[13]:




# In[13]:




# In[ ]:



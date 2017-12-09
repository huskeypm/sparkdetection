
# coding: utf-8

# runner function that will be used for ROC generation/optimization

import util
import cv2
import numpy as np
#import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')
#plt.rcParams['figure.figsize'] = [16,9]
#get_ipython().magic(u'load_ext autoreload')
#get_ipython().magic(u'autoreload 2')
import matchedFilter as mF
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
#import scipy.stats
#import time
import bankDetect as bD
import painter


# In[91]:

# utility to assist in the rotation of the filters
import imutils


# In[93]:

def ReadResizeNormImg(imgName, scale):
    img = cv2.imread(imgName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    normed = resized.astype('float') / float(np.max(resized))
    
    return normed


# # Pre-Processing
# 
# - Rotate myocyte images in gimp s.t. TTs are orthogonal to the x axis
# - Measure the two sarcomere size of the TTs

# # Pass in Arguments

# In[95]:

#root = "/home/AD/dfco222/Desktop/LouchData/"
#imgName = 'Sham_23'
#fileType = '.png'
#WTthreshold=0.045
#Longitudinalthreshold = 0.38
#gamma = 3.


# In[106]:

# distance between the middle of one z-line to the second neighbor of that z-line
filterTwoSarcSize = 25

# designating images used to construct the filters as we won't be testing those
filterDataImages = []

# designating filter roots and filter names. Opting to test with original filter images for now but will test with filters constructed from this data set shortly
WTFilterRoot = "./images/filterImgs/WT/"

LongitudinalFilterRoot = "./images/filterImgs/Longitudinal/"
LongitudinalTwoSarcLengthDict = {"SongWKY_long1":16, "Xie_RV_Control_long2":14, "Xie_RV_Control_long1":16, "Guo2013Fig1C_long1":22}

LossFilterName = "./images/Remodeled_TTs/TT_Idealized_Loss_TruthFilter.png"
LossFilterTwoSarcSize = 28

# parameters for all the different analysis options
applyCLAHE = True
pad = True
plotRawImages = False
binarizeImgs = False
fixWTFilter = True
fixLongFilter = True
plotFilters = False
plotRawFilterResponse = False # MAKE SURE THIS IS OFF FOR LARGE DATA SIZES

# main function
def gimmeStackedHits(imgName, WTthreshold, Longitudinalthreshold, gamma):
  # Read Images and Apply Masks

  # using old code structure
  imgTwoSarcSizesDict = {imgName:'Nonsense'}

  for imgName,imgTwoSarcSize in imgTwoSarcSizesDict.iteritems():
      if imgName not in filterDataImages:
          
          scale = 1. # we're resizing prior to this call
          img = ReadResizeNormImg(imgName, scale)
          combined = img
          
          if applyCLAHE:
              tileSize = filterTwoSarcSize
              combined *= 255
              combined = combined.astype('uint8')
              clahe = util.ApplyCLAHE([combined], (tileSize,tileSize), plot=False)
              #clahe = clahe[0].astype('float') / float(np.max(clahe[0]))
              clahe = clahe[0] / float(np.max(clahe[0]))
              clahe[clahe < 0.02] = 0 # weird issue with clahe adding in background noise
              combined = clahe
              
    
          # routine to pad the image with a border of zeros. This helps with negating issue with shifting nyquist in FFT
          if pad:
              combined = util.PadWithZeros(combined)
    
          imgDim = np.shape(combined)
        
          if plotRawImages:
              plt.figure()
              imshow(combined)
              plt.title(imgName)
              plt.colorbar()
  
  # Read in Filters
  maxResponseDict = {}

  # WT
  WTfilter = util.GenerateWTFilter(WTFilterRoot=WTFilterRoot, filterTwoSarcSize = filterTwoSarcSize)
  if fixWTFilter:
      WTfilter[WTfilter > 0.6] = 0.6
      WTfilter[WTfilter <0.25] = 0
      WTfilter /= np.max(WTfilter)
      WTfilter = WTfilter[20:,1:]

  # Longitudinal
  Longitudinalfilter = util.GenerateLongFilter(LongitudinalFilterRoot,LongitudinalTwoSarcLengthDict,filterTwoSarcLength = filterTwoSarcSize)
  if fixLongFilter:
      Longitudinalfilter[Longitudinalfilter > 0.7] = 0.7
      Longitudinalfilter[Longitudinalfilter < 0.4] = 0
      Longitudinalfilter /= np.max(Longitudinalfilter)
      Longitudinalfilter = Longitudinalfilter[6:13,:-1]

  # Loss
  LossScale = float(filterTwoSarcSize) / float(LossFilterTwoSarcSize)
  Lossfilter = util.GenerateLossFilter(LossFilterName,LossScale)

  # Filter for the punishment of longitudinal regions of WT filter. Improves SNR
  WTPunishFilter = Longitudinalfilter[2:-1,6:13]

  filterDict = {'WT':WTfilter, 'Longitudinal':Longitudinalfilter, 'Loss':Lossfilter, 'WTPunishFilter':WTPunishFilter}

  # finding maximum response of each filter by integrating intensity
  for filterName, myFilter in filterDict.iteritems():
      maxResponseDict[filterName] = np.sum(myFilter)

  if plotFilters:
        for name,Filter in filterDict.iteritems():
              plt.figure()
              imshow(Filter)
              plt.title(name)
              plt.colorbar()


  # # Convolve Each Image with Each Filter

  rotDegrees = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
  display = False
  thresholdDict = {'WT':WTthreshold, 'Longitudinal':Longitudinalthreshold, 'Loss':0.08}
  Result = bD.TestFilters(img,None,None,filterType="TT",
                              display=display,iters=rotDegrees,filterDict=filterDict,thresholdDict=thresholdDict,doCLAHE=False,
                              colorHitsOutName=imgName,
                              label=imgName,
                              saveColoredFig=False,
                              gamma=gamma)

  return Result.coloredImg

# Message printed when program run without arguments 
def helpmsg():
  scriptName= sys.argv[0]
  msg="""
Purpose: A script to aid in the repetitive correlation used for ROC optimization.
 
Usage: Call script from command line with args being: 1. image name and path 2. wild type threshold 3. longitudinal threshold 4. gamma parameter in SNR
"""
  msg+="""
  
 
Notes:

"""
  return msg


if __name__=="__main__":
  import sys
  msg=helpmsg()
  remap = "none"

  if len(sys.argv) != 5:
      raise RuntimeError(msg)


  imgName = str(sys.argv[1])
  WTthresh = float(sys.argv[2])
  Longitudinalthresh = float(sys.argv[3])
  gamma = sys.argv[4]
  result = gimmeStackedHits(imgName, WTthresh, Longitudinalthresh, gamma)
  import matplotlib.pylab as plt
  corr = imgName.split('/')[-1]
  name,filetype = corr.split('.')
  myName = name+'_'+str(WTthresh)+'_'+str(Longitudinalthresh)+'_'+str(gamma)+filetype
  plt.imshow(result)
  plt.gcf().savefig(myName) 

  #print 'successful!'

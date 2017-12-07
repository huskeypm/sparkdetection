
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
import mach
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


#imgTwoSarcSizesDict = {'Sham_P_23':21, 'Sham_M_65':21, 'Sham_D_100':20, 'Sham_23':22,
#                       'Sham_11':21, 'MI_P_8':21, 'MI_P_5':21, 'MI_P_16':21, 'MI_M_46':22,
#                       'MI_M_45':21, 'MI_M_44':21, 'HF_1':21, 'HF_13':21,'MI_D_78':22,
#                       'MI_D_76':21, 'MI_D_73':22,
#                       'HF_5':21 # this myocyte is so bad it may as well be a crap shoot to measure this
#                       }


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

def gimmeStackedHits(imgName, WTthreshold, Longitudinalthreshold, gamma):
  # # Read Images and Apply Masks

  # In[106]:




  # In[107]:

  #imgDict = {}
  #cellContourDict = {}
  #maskDict = {}
  #cellAreaDict = {}

  # using old code structure
  imgTwoSarcSizesDict = {imgName:'Nonsense'}

  for imgName,imgTwoSarcSize in imgTwoSarcSizesDict.iteritems():
      if imgName not in filterDataImages:
          
          scale = 1. # we're resizing prior to this call
          img = ReadResizeNormImg(imgName, scale)
    
          # read/construct mask
          #mask = ReadResizeNormImg(root+imgName+'_mask'+fileType, scale)
          #mask[mask<1.0] = 0
    
          # apply mask
          #combined = img * mask
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
    
          # add image to dictionary
          #imgDict[imgName] = combined
          imgDim = np.shape(combined)
        
          ''' 
          Measuring cell area based on contouring. Doing this so that it is easy to adapt the code once 
          edge detection routine that Pete has in mind is in place.
          '''
          #mask = combined.copy()
          #mask[mask > 0.02] = 1 
          #plt.figure()
          #imshow(mask)
          #contourImg = mask.astype('uint8')
  
          #contours, hierarchy = cv2.findContours(contourImg,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
          #cellCont = max(contours, key=cv2.contourArea)
  
          #cellContourDict[imgName] = cellCont
        
          # measure area for cell
          #cellAreaDict[imgName] = cv2.contourArea(cellCont)
        
          # create mask from the contoured image
          #mask = np.zeros(imgDim)
          #print type(mask)
          #cv2.drawContours(mask,[cellCont],-1,1,-1)
          
          # using trick to 'ignore' the NaNs in the averaging so it doesn't affect WT striation averaging
          #mask[mask == 0] = np.nan
          #maskDict[imgName] = mask        
          #plt.figure()
          #imshow(mask)
          
          if plotRawImages:
              plt.figure()
              imshow(combined)
              plt.title(imgName)
              plt.colorbar()
  

  # In[108]:

  #img = img[50:200,200:600]
  #imshow(img)


  # # Read in Filters

  # In[109]:

  maxResponseDict = {}

  # WT
  WTfilter = util.GenerateWTFilter(WTFilterRoot=WTFilterRoot, filterTwoSarcSize = filterTwoSarcSize)
  if fixWTFilter:
      #WTfilter = WTfilter[30:,:]
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
  #filterDict = {'WT':WTfilter, 'Longitudinal':Longitudinalfilter, 'WTPunishFilter':WTPunishFilter}

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

  # ## <font color=red> Now attempting to implement Ryan's bankDetect.py Filter Rotation Routine </font>

  # In[110]:

  rotDegrees = [-20, -15, -10, -5, 0, 5, 10, 15, 20] # these are really subtle angles
  display = False
  thresholdDict = {'WT':WTthreshold, 'Longitudinal':Longitudinalthreshold, 'Loss':0.08}
  Result = bD.TestFilters(img,None,None,filterType="TT",
                              display=False,iters=rotDegrees,filterDict=filterDict,thresholdDict=thresholdDict,doCLAHE=False,
                              colorHitsOutName=imgName,
                              label=imgName,
                              saveColoredFig=False,
                              gamma=gamma)


  # In[111]:

  #imshow(Result.coloredImg)
  return Result.coloredImg

# In[101]:




# In[101]:




# In[101]:
#
# Message printed when program run without arguments 
#
def helpmsg():
  scriptName= sys.argv[0]
  msg="""
Purpose: A script to aid in the repetitive correlation used for ROC optimization.
 
Usage: Call script from command line with args being: 1. image name and path 2. wild type threshold 3. longitudinal threshold 3. gamma parameter in SNR
"""
  msg+="""
  
 
Notes:

"""
  return msg


if __name__=="__main__":
  import sys
  msg=helpmsg()
  remap = "none"

  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  ### PARAMS 
#  for i,arg in enumerate(sys.argv):
#    imgName = 

  ### RUNS
#  for i,arg in enumerate(sys.argv):
#    if(arg=="-run"):

  imgName = str(sys.argv[1])
  #print imgName
  WTthresh = sys.argv[2]
  Longitudinalthresh = sys.argv[3]
  gamma = sys.argv[4]
  result = gimmeStackedHits(imgName, WTthresh, Longitudinalthresh, gamma)
  print 'successful!'

from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.mlab as mlab
import cv2
from scipy.misc import toimage
from scipy.ndimage.filters import *
import matchedFilter as mF
import imutils
from imtools import *
from matplotlib import cm
def padWithZeros(array, padwidth, iaxis, kwargs):
    array[:padwidth[0]] = 0
    array[-padwidth[1]:]= 0
    return array




# Need to be careful when cropping image
def correlateThresher(myImg, myFilter1,  threshold = 190, cropper=[25,125,25,125],iters = [0,30,60,90],  fused = True, printer = True):
    # PKH 
    correlated = []

    # Ryan ?? equalized image?
    clahe99 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    cl1 = clahe99.apply(myImg)
    cv2.imwrite('clahe_99.jpg',cl1)
    adapt99 = ReadImg('clahe_99.jpg')

    for i, val in enumerate(iters):
      # ????????
      tracker = np.copy(adapt99)
      
      # pad/rotate 
      dims = np.shape(myFilter1)
      diff = np.min(dims)
      paddedFilter = np.lib.pad(myFilter1,diff,padWithZeros)
      rotatedFilter = imutils.rotate(paddedFilter,-val)
      rF = np.copy(rotatedFilter)
    
      #if printer:   
        #plt.figure()
        #plt.title("UNROTATED IMAGE")

      # matched filtering 
      hXtal = mF.matchedFilter(tracker,rF)
      
      # crop/rotate image 
      rotated = imutils.rotate(hXtal,(-val-1))[cropper[0]:cropper[1],cropper[2]:cropper[3]]
      unrotated = hXtal[cropper[0]:cropper[1],cropper[2]:cropper[3]]

      # store data 
      correlated.append(rotated) 
    
      # store outputs
      # Ryan: my general preference is to have one line per operation for clarity
      if fused:
        tag = "fusedCorrelated"
      else: 
        tag = "bulkCorrelated"
      #  toimage(imutils.rotate(hXtal,(-val-1))[cropper[0]:cropper[1],cropper[2]:cropper[3]]).save('bulkCorrelated_{}.png'.format(val))
      #  toimage(hXtal[cropper[0]:cropper[1],cropper[2]:cropper[3]]).save('bulkCorrelated_Not_rotated_back{}.png'.format(val))

      # save
      toimage(rotated).save(tag+'_{}.png'.format(val))
      toimage(unrotated).save(tag+'_Not_rotated_back{}.png'.format(val))


    return correlated



def paintME(myImg, myFilter1,  threshold = 190, cropper=[24,129,24,129],iters = [0,30,60,90], fused =True):
  correlateThresher(myImg, myFilter1,  threshold, cropper,iters, fused, False)
  for i, val in enumerate(iters):
 
    if fused:
      palette = cm.gray
      palette.set_bad('m', 1.0)
      placer = ReadImg('fusedCorrelated_{}.png'.format(val))
    else:
      palette = cm.gray
      palette.set_bad('b', 1.0)
      placer = ReadImg('bulkCorrelated_{}.png'.format(val))
    plt.figure()

    #print "num maxes", np.shape(np.argwhere(placer>threshold))
    Zm = np.ma.masked_where(placer > threshold, placer)
    fig, ax1 = plt.subplots()
    plt.axis("off")
    im = ax1.pcolormesh(Zm, cmap=palette)
    plt.title('Correlated_Angle_{}'.format(val))
    plt.savefig('falseColor_{}.png'.format(val))
    plt.axis('equal')
    
                

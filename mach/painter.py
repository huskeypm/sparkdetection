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



class empty:pass

# Need to be careful when cropping image
def correlateThresher(myImg, myFilter1,  cropper=[25,125,25,125],
                      iters = [0,30,60,90],  fused = True, printer = True):
    # PKH 
    correlated = []

    # Ryan ?? equalized image?
    clahe99 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    cl1 = clahe99.apply(myImg)
    cv2.imwrite('clahe_99.jpg',cl1)
    adapt99 = ReadImg('clahe_99.jpg')

    for i, val in enumerate(iters):
      result = empty()
      # ????????
      tracker = np.copy(adapt99)
      
      # pad/rotate 
      dims = np.shape(myFilter1)
      diff = np.min(dims)
      paddedFilter = np.lib.pad(myFilter1,diff,padWithZeros)
      rotatedFilter = imutils.rotate(paddedFilter,-val)
      rF = np.copy(rotatedFilter)
    

      # matched filtering 
      hXtal = mF.matchedFilter(tracker,rF)
      
      # crop/rotate image 
      # if filter is being rotated, I don't think we need to rotted the correlated image 
      #rotated = imutils.rotate(hXtal,(-val-1)) # [cropper[0]:cropper[1],cropper[2]:cropper[3]]
      unrotated = hXtal # [cropper[0]:cropper[1],cropper[2]:cropper[3]]

      # spot check results
      hit = np.max(unrotated) 
      hitLoc = np.argmax(unrotated) 
      hitLoc =np.unravel_index(hitLoc,np.shape(unrotated))

      # store

      daTitle = "rot %f "%val + "hit %f "%hit + str(hitLoc)
      print daTitle
      if printer:   
        plt.figure()
        plt.subplot(1,3,1)
        plt.title(daTitle)
        plt.imshow(tracker,cmap="gray")   
        plt.subplot(1,3,2)
        plt.imshow(rF,cmap="gray")   
        
        plt.subplot(1,3,3)
        plt.imshow(unrotated)   


      # store data 
      result.corr = unrotated 
      result.hit = hit
      result.hitLoc = hitLoc
      correlated.append(result) 
    
      # store outputs
      # Ryan: my general preference is to have one line per operation for clarity
      if fused:
        tag = "fusedCorrelated"
      else: 
        tag = "bulkCorrelated"
      #  toimage(imutils.rotate(hXtal,(-val-1))[cropper[0]:cropper[1],cropper[2]:cropper[3]]).save('bulkCorrelated_{}.png'.format(val))
      #  toimage(hXtal[cropper[0]:cropper[1],cropper[2]:cropper[3]]).save('bulkCorrelated_Not_rotated_back{}.png'.format(val))

      # save
      #toimage(rotated).save(tag+'_{}.png'.format(val))
      toimage(unrotated).save(tag+'_Not_rotated_back{}.png'.format(val))


    return correlated

import util 
import util2
def StackHits(correlated,threshold,iters,
              display=False,rescaleCorr=False,doKMeans=True):
    maskList = []

    for i, iteration in enumerate(iters):
        print "iter", iteration
        #maskList.append(makeMask(threshold,'fusedCorrelated_Not_rotated_back{}.png'.format(iteration)))

        # RYAN
        #maskList.append((util2.rotater(util2.makeMask(threshold,imgName='fusedCorrelated_{}.png'.format(iteration)),iteration)))
        #imgName='fusedCorrelated_{}.png'.format(iteration)
        #daMask = util2.makeMask(threshold,imgName=imgName)

        print "WARNING: please dig into why find_centers fails from time to time (or look into more robust clustering routine)"
        # Ryan - I don't think this renormalization is appropriate
        # as it will artificially inflate 'bad' correlation hits
        corr_i = correlated[i].corr           
        if rescaleCorr:
           img =  util.renorm(corr_i)
        else: 
           img=corr_i
        #print img

        # routine for identifying 'unique' hits
        #performed on 'unrotated' images 
        daMask = util2.makeMask(threshold,img = img,doKMeans=doKMeans)
        if display:
          plt.figure()
          plt.subplot(1,2,1)
          plt.imshow(img)            
          plt.subplot(1,2,2)
          plt.imshow(daMask)

        # i don't think this should be rotated 
        #maskList.append((util2.rotater(daMask,iteration)))
        maskList.append(daMask)
    #print maskList

    myList  = np.sum(maskList, axis =0)
    if display: 
      plt.figure()
      plt.imshow(myList)
    return myList





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
    
                

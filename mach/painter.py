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

# Need to be careful when cropping image
def correlateThresher(myImg, myFilter1,  threshold = 190, cropper=[25,125,25,125],iters = [0,30,60,90], printer = True):
    for i, val in enumerate(iters):
      clahe99 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
      cl1 = clahe99.apply(myImg)
      cv2.imwrite('clahe_99.jpg',cl1)
      adapt99 = ReadImg('clahe_99.jpg')
      tracker = np.copy(adapt99)
      dst = imutils.rotate(tracker,(val))
      dst1 = np.copy(dst)

      if printer:   
        plt.figure()
        myplot(tracker[cropper[0]:cropper[1],cropper[2]:cropper[3]])
        plt.title("UNROTATED IMAGE")
      hXtal = mF.matchedFilter(dst1,myFilter1)
      toimage(imutils.rotate(hXtal,(-val-1))[cropper[0]:cropper[1],cropper[2]:cropper[3]]).save('fusedCorrelated_{}.png'.format(val))
      toimage(hXtal[cropper[0]:cropper[1],cropper[2]:cropper[3]]).save('fusedCorrelated_Not_rotated_back{}.png'.format(val))



def paintME(myImg, myFilter1,  threshold = 190, cropper=[24,129,24,129],iters = [0,30,60,90]):
  correlateThresher(myImg, myFilter1,  threshold = 190, cropper=[24,129,24,129],iters = [0,30,60,90])
  for i, val in enumerate(iters):
 
    placer = ReadImg('fusedCorrelated_{}.png'.format(val))
    
    plt.figure() 
    palette = cm.gray
    palette.set_bad('m', 1.0)
    
    print "num maxes", np.shape(np.argwhere(placer>threshold))
    Zm = np.ma.masked_where(placer > threshold, placer)
    fig, ax1 = plt.subplots()
    plt.axis("off")
    im = ax1.pcolormesh(Zm, cmap=palette)
    plt.title('Correlated_Angle_{}'.format(val))
    plt.savefig('falseColor_{}.png'.format(val))
    plt.axis('equal')
    
                

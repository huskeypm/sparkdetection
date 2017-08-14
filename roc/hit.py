"""
### Evaluate hit:
- select points within 3 pixels of max
- integrate intensity
- collect annulus around maxima that does not include 'maximum area' and compute mean
- mask out region in original thresholded image
- store intensity/annulus_mean in output image
"""
import numpy as np 

import sys
sys.path.append("../mach")
import util 

# image - original image
# superThresh - image with -subthreshold' parts set to 0
# hits - array indicating points that pass
def EvaluateNextHit(image,superThresh,hits,
                    innerMargin=7,outerMargin=13,threshold=0,
                    verbose =True):
    ## Get top hit by intensity 
    flat = np.ndarray.flatten(superThresh)
    sidx = np.argsort( flat )[::-1]
    top = np.unravel_index(sidx[0], np.shape(superThresh))
    #print top
    #print "sidx", sidx
    ## Get signal/sidelobe ratio
    #signalInner,integratedValue,area = util.GetRegion(image,top,margin=7)
    #signalOuter,integratedValue,area = util.GetRegion(image,top,margin=13)
    annulus,interior = util.GetAnnulus(
      image,top,innerMargin=innerMargin,outerMargin=outerMargin)
    mean,stddev = np.mean(annulus), np.std(annulus)
    integrated = np.sum(interior)
    #print "top", top
    #print "shape(interior)", np.shape(interior)
    #print "interior", interior
    #print "np.shape(annulus)", np.shape(annulus)
    #print "integrated", integrated
    #print "mean", mean

    value = integrated / mean

    ## mask image 
    util.MaskRegion(superThresh,top,outerMargin,value=0)
    #print "value, you nerd", value
    ## tag solutions
    isHit = value > threshold
    if isHit:
      util.MaskRegion(hits,top,3,value=1)
    if verbose:
      print top, value, isHit
    return isHit



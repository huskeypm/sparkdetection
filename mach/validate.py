import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy.misc import toimage
import imutils
import imtools as it
import numpy as np
import painter as Paint
root = "/net/share/shared/papers/nanoporous/images/"
# Raw data with fused pores
def ValidateRot(
    rot = 30
     ):
    rawFused = cv2.imread(root+'fusedCroppedCLAHE.png')
    fusedFilter = cv2.imread(root+'fusedBase.png')
    rawFused = cv2.cvtColor(rawFused, cv2.COLOR_BGR2GRAY)
    fusedFilter = cv2.cvtColor(fusedFilter, cv2.COLOR_BGR2GRAY)

    iters = np.linspace(0,90,10)

    # create image with filter in it (for debugging) 
    testRaw = np.zeros_like(rawFused)

    # rot filter inside 'test' image     
    rotFilter = imutils.rotate(fusedFilter,-rot)
    dims = np.shape(rotFilter)
    offset = 40
    testRaw[offset:(offset+dims[0]),offset:(offset+dims[1])] = rotFilter
    #imshow(testRaw)

    # correlate against bank of rotated filters 
    correlated = Paint.correlateThresher(testRaw,fusedFilter, iters=iters,fused=True,printer=False)

    best=-1e9
    besti = None
    for i,result in enumerate( correlated ):
        if result.hit>best:
            best = result.hit
            besti =i
    print "Best Match ", iters[besti], " Truth ", rot   
    margin = np.abs((iters[1]-iters[0])/2.)
    absdiff = np.abs(iters[besti]-rot)
    assert(absdiff < margin),"FAIL"
    if (absdiff < margin):
        print "MATCH!"
    else:
        print "FAIL!!"


ValidateRot()   

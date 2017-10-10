import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy.misc import toimage
import imutils
import imtools as it
import numpy as np
import painter as Paint
root = "/net/share/shared/papers/nanoporous/images/"
# Contains signal embedded in real data
# slightly more difficult test 
def ValidateRot2(
    rot = 30,
    offset = 40, # xy offset 
    scale = 0.20,# scale for embedded signal 
    iters = np.linspace(0,90,10),
    display=False
     ):
    rawFused = cv2.imread(root+'fusedCroppedCLAHE.png')
    fusedFilter = cv2.imread(root+'fusedBase.png')
    rawFused = cv2.cvtColor(rawFused, cv2.COLOR_BGR2GRAY)
    fusedFilter = cv2.cvtColor(fusedFilter, cv2.COLOR_BGR2GRAY)



    # rot filter inside 'test' image     
    rotFilter = imutils.rotate(fusedFilter,-rot)
    dims = np.shape(rotFilter)
    testRaw = np.array(rawFused,dtype=np.float) 
    #testRaw = rawFused
    testRaw[offset:(offset+dims[0]),offset:(offset+dims[1])] += scale * rotFilter
    print np.max(testRaw)   
    testRaw[np.where(testRaw>255)]=255 
    testRaw = np.array(testRaw,dtype=np.uint8)
    #testRaw = np.array(testRaw,dtype="uint8") 
    #imshow(testRaw)

    # correlate against bank of rotated filters 
    correlated = Paint.correlateThresher(testRaw,fusedFilter, iters=iters,fused=True,printer=display)

    TestBest(correlated,rot,iters) 
    return correlated

def TestBest(correlated,truthRot,iters):
    best=-1e9
    besti = None
    for i,result in enumerate( correlated ):
        if result.hit>best:
            best = result.hit
            besti =i
    print "Best Match ", iters[besti], " Truth ", truthRot 
    margin = np.abs((iters[1]-iters[0])/2.)
    absdiff = np.abs(iters[besti]-truthRot)
    assert(absdiff < margin),"FAIL"
    if (absdiff < margin):
        print "MATCH!"
    else:
        print "FAIL!!"

# Creates test data consisting only of signal (best case scenario) 
def ValidateRot(
    rot = 30,
    display=False
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
    correlated = Paint.correlateThresher(testRaw,fusedFilter, iters=iters,fused=True,printer=display)

    TestBest(correlated,rot,iters) 


#!/usr/bin/env python
import sys
##################################
#
# Revisions
#       10.08.10 inception
#
##################################

#
# ROUTINE  
#


#
# Message printed when program run without arguments 
#
def helpmsg():
  scriptName= sys.argv[0]
  msg="""
Purpose: 
 
Usage:
"""
  msg+="  %s -validation" % (scriptName)
  msg+="""
  
 
Notes:

"""
  return msg

#
# MAIN routine executed when launching this script from command line 
#
if __name__ == "__main__":
  import sys
  msg = helpmsg()
  remap = "none"

  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  #fileIn= sys.argv[1]
  #if(len(sys.argv)==3):
  #  1
  #  #print "arg"

  # Loops over each argument in the command line 
  for i,arg in enumerate(sys.argv):
    # calls 'doit' with the next argument following the argument '-validation'
    if(arg=="-validation"):
      #arg1=sys.argv[i+1] 
      ValidateRot()
      quit()
  





  raise RuntimeError("Arguments not understood")





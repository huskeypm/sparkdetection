"""
Finds all hits for rotated filter 
"""
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class empty:pass

import cv2
import painter as Paint 
import numpy as np 
import matplotlib.pylab as plt 


def DetectFilter(dataSet,mf,threshold,iters,display=False,sigma_n=1.,mode=None):
  # store
  result = empty()
  result.threshold = threshold
  result.mf= mf

  if mode=="fused":
    fused = True
  elif mode=="bulk":
    fused = False
  else:
    raise RuntimeError("mode unkwnoen") 
    
  # do correlations across all iter
  result.correlated = Paint.correlateThresher(
     dataSet,result.mf, threshold = result.threshold, iters=iters,fused=fused,
     printer=display,sigma_n=sigma_n,
     label=mode)

  # 
  snrs = [] 
  for i, resulti in enumerate(result.correlated):
    maxSNR = np.max( resulti.snr) 
    snrs.append( maxSNR )              
  result.snrs = np.array( snrs)
  result.iters = iters 
  print result.snrs
  print result.iters 
  
  # stack hits to form 'total field' of hits
  result.stackedHits = Paint.StackHits(
    result.correlated,result.threshold,iters,doKMeans=False, display=False)#,display=display)
    
  return result


def GetHits(aboveThresholdPoints):
  ## idenfity hits (binary)  
  mylocs =  np.zeros_like(aboveThresholdPoints.flatten())
  hits = np.argwhere(aboveThresholdPoints.flatten()>0)
  mylocs[hits] = 1
  
  #locs = mylocs.reshape((100,100,1))
  dims = np.shape(aboveThresholdPoints)  
  #locs = mylocs.reshape((dims[0],dims[1],1))  
  locs = mylocs.reshape(dims)  
  #print np.sum(locs)    
  #print np.shape(locs)  
  #plt.imshow(locs)  
  #print np.shape(please)
  #Not sure why this is needed  
  #zeros = np.zeros_like(locs)
  #please = np.concatenate((locs,zeros,zeros),axis=2)
  #print np.shape(please)
  return locs

# color red channel 
def ColorChannel(Img,stackedHits,chIdx=0):  
    locs = GetHits(stackedHits)   
    chFloat =np.array(Img[:,:,chIdx],dtype=np.float)
    #chFloat[10:20,10:20] += 255 
    chFloat+= 255*locs  
    chFloat[np.where(chFloat>255)]=255
    Img[:,:,chIdx] = np.array(chFloat,dtype=np.uint8)  



# red - entries where hits are to be colored (same size as rawOrig)
# will label in rawOrig detects in the 'red' channel as red, etc 
def colorHits(rawOrig,red=None,green=None,outName=None):
  dims = np.shape(rawOrig)  
  
  # make RGB version of data   
  Img = np.zeros([dims[0],dims[1],3],dtype=np.uint8)
  scale = 0.5  
  Img[:,:,0] = scale * rawOrig
  Img[:,:,1] = scale * rawOrig
  Img[:,:,2] = scale * rawOrig
    

  
  if isinstance(red, (list, tuple, np.ndarray)): 
    ColorChannel(Img,red,chIdx=0)
  if isinstance(green, (list, tuple, np.ndarray)): 
    ColorChannel(Img,green,chIdx=1)    
    
  plt.figure()  
  plt.subplot(1,2,1)
  plt.imshow(rawOrig,cmap='gray')
  plt.subplot(1,2,2)
  plt.imshow(Img)  
  if outName!=None:
    plt.gcf().savefig(outName,dpi=300)
    


# main engine 

def TestFilters(testDataName,fusedFilterName,bulkFilterName,fusedThresh=60,bulkThresh=50,
                subsection=None,
                display=False,
                sigma_n = 1., 
                iters = [0,10,20,30,40,50,60,70,80,90], 
                outName="test.png"):

    
    # load data against which filters are tested
    testData = cv2.imread(testDataName)
    testData = cv2.cvtColor(testData, cv2.COLOR_BGR2GRAY)
    if isinstance(subsection, (list, tuple, np.ndarray)): 
      testData = testData[subsection[0]:subsection[1],subsection[2]:subsection[3]]

    # load fused filter
    fusedFilter = cv2.imread(fusedFilterName)
    fusedFilter = cv2.cvtColor(fusedFilter, cv2.COLOR_BGR2GRAY)

    # load bulk filter 
    bulkFilter = cv2.imread(bulkFilterName)
    bulkFilter = cv2.cvtColor(bulkFilter, cv2.COLOR_BGR2GRAY)
    #unitBulk = fusedReal[305:327,318:358]

    fusedPoreResult = DetectFilter(testData,fusedFilter,fusedThresh,
                                   iters,display=display,sigma_n=sigma_n,mode="fused")
    bulkPoreResult = DetectFilter(testData,bulkFilter,bulkThresh,
                                  iters,display=display,sigma_n=sigma_n,mode="bulk")
    colorHits(testData,red=bulkPoreResult.stackedHits,green=fusedPoreResult.stackedHits,
                 outName=outName)

    return fusedPoreResult, bulkPoreResult 


def TestTrueData():
  root = "/net/share/shared/papers/nanoporous/images/"
  img1 = '/home/AD/srbl226/spark/sparkdetection/roc/clahe_Best.jpg'
  img2 = root+"full.png"
  dummy = TestFilters(
    img1, # testData
    root+'fusedBase.png',         # fusedfilter Name
    root+'bulkCellTEM.png',        # bulkFilter name
    #subsection=[200,400,200,500],   # subsection of testData
    subsection=[200,400,200,500],   # subsection of testData
    fusedThresh = 6.,  
    bulkThresh = 6., 
    outName = "filters_on_pristine.png",
    display=False   
  )  
  
  
  
  
  
  
  
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
      TestTrueData() 
      quit()






  raise RuntimeError("Arguments not understood")



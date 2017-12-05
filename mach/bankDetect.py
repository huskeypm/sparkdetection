"""
Finds all hits for rotated filter 
"""
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class empty:pass

import cv2
import painter 
import numpy as np 
import matplotlib.pylab as plt 


def DetectFilter(dataSet,mf,threshold,iters,display=False,sigma_n=1.,
                 label=None,filterMode=None,useFilterInv=False,scale=1.,
                 doCLAHE=True,filterType="Pore"):

  # store
  result = empty()
  # difference for TT routines these are now dictionaries
  result.threshold = threshold
  result.mf= mf

  if filterType == "Pore":
    # do correlations across all iter
    result.correlated = painter.correlateThresher(
       dataSet,result.mf, threshold = result.threshold, iters=iters,
       printer=display,sigma_n=sigma_n,
       scale=scale,
       filterMode=filterMode,
       useFilterInv=useFilterInv,
       label=label,
       doCLAHE=doCLAHE)

    # 
    snrs = [] 
    for i, resulti in enumerate(result.correlated):
      maxSNR = np.max( resulti.snr) 
      snrs.append( maxSNR )              
    result.snrs = np.array( snrs)
    result.iters = iters 
  
    # stack hits to form 'total field' of hits
    result.stackedHits = painter.StackHits(
      result.correlated,result.threshold,iters,doKMeans=False, display=False)#,display=display)
    
  elif filterType == "TT":
    result.correlated = painter.correlateThresherTT(
       dataSet,result.mf, thresholdDict=result.threshold,iters=iters,doCLAHE=doCLAHE)

    # stack filter hits
    result.stackedHits = painter.StackHits(result.correlated,threshold,iters,display=display,doKMeans=False,
                                           filterType="TT")
  
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
def colorHits(rawOrig,red=None,green=None,outName=None,label="",plotMe=True):
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

  if plotMe:
    plt.figure()  
    plt.subplot(1,2,1)
    plt.title("Raw data (%s)"%label)
    plt.imshow(rawOrig,cmap='gray')
    plt.subplot(1,2,2)
    plt.title("Marked") 
    plt.imshow(Img)  
  if outName!=None:
    plt.tight_layout()
    plt.gcf().savefig(outName,dpi=300)
  else:
    return Img  


# main engine 

def TestFilters(testDataName,fusedFilterName,bulkFilterName,
                fusedThresh=60,bulkThresh=50,
                subsection=None,
                display=False,
                colorHitsOutName=None,
                sigma_n = 1., 
                iters = [0,10,20,30,40,50,60,70,80,90], 
                scale=1.0,      
                useFilterInv=False,
                label="test",
                filterType="Pore",
                filterDict=None, thresholdDict=None,
                doCLAHE=True,saveColoredFig=True):       

    if filterType == "Pore":
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


      fusedPoreResult = DetectFilter(testData,fusedFilter,fusedThresh,
                                     iters,display=display,sigma_n=sigma_n,
                                     filterMode="fused",label=label,
                                     scale=scale,
                                     useFilterInv=useFilterInv)
      bulkPoreResult = DetectFilter( testData,bulkFilter,bulkThresh,
                                     iters,display=display,sigma_n=sigma_n,
                                     filterMode="bulk",label=label,
                                     scale=scale,
                                     useFilterInv=useFilterInv)

      if colorHitsOutName!=None: 
        colorHits(testData,
                red=bulkPoreResult.stackedHits,
                green=fusedPoreResult.stackedHits,
                label=label,
                outName=colorHitsOutName)                       

      return fusedPoreResult, bulkPoreResult 

    elif filterType == "TT":
      # really ugly adaptation but I'm storing both loss and longitudinal filters in fusedFilterName
      # WTfilter and WT punishment filter in bulkFilterName
      #WTfilter = bulkFilterName['WT']
      #WTpunishment = bulkFilterName['WTPunishFilter']
      #Lossfilter = fusedFilterName['Loss']
      #Longfilter = fusedFilterName['Longitudinal']

      # utilizing runner functions to produce stacked images
      resultContainer = DetectFilter(testDataName,filterDict,thresholdDict,iters,display=display,sigma_n=sigma_n,
                                     filterType="TT",doCLAHE=doCLAHE)

      if colorHitsOutName != None and saveColoredFig:
        colorImg = testDataName * 255
        colorImg = colorImg.astype('uint8')
        colorHits(colorImg, red=resultContainer.stackedHits.WT, green=resultContainer.stackedHits.Long,
                  #blue=resultContainer.stackedHits.Loss,
                  label=label,outName=colorHitsOutName)
      elif colorHitsOutName != None and not saveColoredFig:
        colorImg = testDataName * 255
        colorImg = colorImg.astype('uint8')
        resultContainer.coloredImg = colorHits(colorImg, red=resultContainer.stackedHits.WT, green=resultContainer.stackedHits.Long,
                                               label=label,outName=None, plotMe=False)

      return resultContainer

    else:
      raise RuntimeError, "Filtering type not understood"


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
    colorHitsoutName = "filters_on_pristine.png",
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



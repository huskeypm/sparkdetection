"""
Purpose of this program is to optimize the threshold parameters 
to cleanly pick out data features
"""
import cv2
import bankDetect as bD
import numpy as np
import matplotlib.pylab as plt


# fused Pore 
class empty():pass
root = "./testimages/"
sigma_n = 22. # based on Ryan's data 
fusedThresh = 1000.
bulkThresh = 1050. 

def Score(positiveHits,negativeHits,truthName,display=True):
    # read in 'truth image' 
    truthMarked = cv2.imread(truthName)
    truthMarked=cv2.cvtColor(truthMarked, cv2.COLOR_BGR2GRAY)
    truthMarked= np.array(truthMarked> 0, dtype=np.float)
    #imshow(fusedMarked)

    # positive hits 
    positiveMasked = np.array(positiveHits > 0, dtype=np.float)
    if display:
      plt.figure()    
      plt.subplot(1,3,1)
      plt.imshow(positiveMasked)
      plt.subplot(1,3,2)
      plt.imshow(truthMarked)
      plt.subplot(1,3,3)
      composite = 2.*truthMarked + positiveMasked
      plt.imshow(composite)
    #plt.imsave("Test.png",composite)
    positiveScoreImg = truthMarked*positiveMasked
    positiveScore = np.sum(positiveScoreImg)/np.sum(truthMarked)

    # negative hits 
    negativeMasked = np.array(negativeHits > 0, dtype=np.float)
    if display: 
      plt.figure()    
      plt.subplot(1,3,1)
      plt.imshow(negativeMasked)
      plt.subplot(1,3,2)
      plt.imshow(truthMarked)
      plt.subplot(1,3,3)
      composite = 2.*truthMarked + negativeMasked
      plt.imshow(composite)
    negativeScoreImg = truthMarked*negativeMasked
    negativeScore = np.sum(negativeScoreImg)/np.sum(truthMarked)

    return positiveScore, negativeScore

def TestParams(fusedThresh=1000.,bulkThresh=1050.,sigma_n=1.,display=False):
    ### Fused pore
    testCase = empty()
    testCase.name = root + 'clahe_Best.jpg'
    testCase.subsection=[340,440,400,500]
    #daImg = cv2.imread(testCase.name)
    #cut = daImg[testCase.subsection[0]:testCase.subsection[1],testCase.subsection[2]:testCase.subsection[3]]
    #imshow(cut)

    fusedPoreResult, bulkPoreResult = bD.TestFilters(
      testCase.name, # testData
      root+'fusedCellTEM.png',         # fusedfilter Name
      root+'bulkCellTEM.png',        # bulkFilter name
      subsection=testCase.subsection, #[200,400,200,500],   # subsection of testData
      fusedThresh = fusedThresh,#.25,
      bulkThresh = bulkThresh, #.5,
      #label = "opt.png",
      sigma_n = sigma_n,
      iters = [30],
      display=display
    )        

    fusedPS, bulkNS= Score(fusedPoreResult.stackedHits,bulkPoreResult.stackedHits,"testimages/fusedMarked.png",display=display)

    ### Bulk pore
    testCase = empty()
    testCase.name = root+'clahe_Best.jpg'
    testCase.subsection=[250,350,50,150]
    #daImg = cv2.imread(testCase.name)
    #cut = daImg[testCase.subsection[0]:testCase.subsection[1],testCase.subsection[2]:testCase.subsection[3]]
    #imshow(cut)

    if 1: 
      fusedPoreResult, bulkPoreResult = bD.TestFilters(
      testCase.name,
      root+'fusedCellTEM.png',         # fusedfilter Name
      root+'bulkCellTEM.png',        # bulkFilter name
      subsection=testCase.subsection, #[200,400,200,500],   # subsection of testData
      fusedThresh = fusedThresh,#80.,
      bulkThresh = bulkThresh,#130.,
      label = "filters_on_pristine.png",
      sigma_n=sigma_n,
      iters = [5],
      display=display
     )        
    
    
    bulkPS, fusedNS = Score(bulkPoreResult.stackedHits,fusedPoreResult.stackedHits,"testimages/bulkMarked.png",display=display)   
    
    ## 
    return fusedPS,bulkNS,bulkPS,fusedNS

def AnalyzePerformanceData(df,tag='bulk'):

    #plt.figure()
    threshID=tag+'Thresh'
    result = df.sort_values(by=[threshID], ascending=[1])

    plt.title(threshID+" threshold")
    plt.scatter(df[threshID], df[tag+'PS'],label=tag+"/positive",c='b')
    plt.scatter(df[threshID], df[tag+'NS'],label=tag+"/negative",c='r')


    fig, ax = plt.subplots()
    ax.set_title("ROC")
    ax.scatter(df[tag+'NS'],df[tag+'PS'])

    i =  np.int(0.45*np.shape(result)[0])
    numbers = np.arange( np.shape(result)[0])
    numbers = numbers[::25]
    #numbers = [i]
    for i in numbers:
        #print i
        thresh= result[threshID].values[i]
        ax.scatter(result[tag+'NS'].values[i],result[tag+'PS'].values[i],c="r")
        loc = (result[tag+'NS'].values[i],0.1+result[tag+'PS'].values[i])
        ax.annotate("%4.1f"%thresh, loc)

import pandas as pd

def Assess(
  fusedThreshes = np.linspace(800,1100,10), 
  bulkThreshes = np.linspace(800,1100,10), 
  sigma_n = 1.,
  display=False
  ):
  
  # create blank dataframe
  df = pd.DataFrame(columns = ['fusedThresh','bulkThresh','fusedPS','bulkNS','bulkPS','fusedNS'])
  
  # iterate of thresholds
  for i,fusedThresh in enumerate(fusedThreshes):
    for j,bulkThresh in enumerate(bulkThreshes):
      fusedPS,bulkNS,bulkPS,fusedNS = TestParams(fusedThresh=fusedThresh,bulkThresh=bulkThresh,sigma_n=sigma_n,display=display)
      raw_data =  {\
       'fusedThresh': fusedThresh,
       'bulkThresh': bulkThresh,
       'fusedPS': fusedPS,
       'bulkNS': bulkNS,
       'bulkPS': bulkPS,
       'fusedNS': fusedNS}
      #print raw_data
      dfi = pd.DataFrame(raw_data,index=[0])#columns = ['fusedThresh','bulkThresh','fusedPS','bulkNS','bulkPS','fusedNS'])
      df=df.append(dfi)

  # store in hdf5 file
  hdf5Name = "optimizer.h5"
  print "Printing " , hdf5Name 
  df.to_hdf(hdf5Name,'table', append=False)
  
  return df,hdf5Name     
  
  


#!/usr/bin/env python
import sys
##################################
#
# Revisions
#       10.08.10 inception
#
##################################


#
# Message printed when program run without arguments 
#
def helpmsg():
  scriptName= sys.argv[0]
  msg="""
Purpose: 
 
Usage:
"""
  msg+="  %s -optimize" % (scriptName)
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
    if(arg=="-optimize"):
      Assess()
      quit()
  





  raise RuntimeError("Arguments not understood")





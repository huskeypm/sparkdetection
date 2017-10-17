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

def Score(positiveHits,negativeHits,truthName):
    # read in 'truth image' 
    truthMarked = cv2.imread(truthName)
    truthMarked=cv2.cvtColor(truthMarked, cv2.COLOR_BGR2GRAY)
    truthMarked= np.array(truthMarked> 0, dtype=np.float)
    #imshow(fusedMarked)

    # positive hits 
    positiveMasked = np.array(positiveHits > 0, dtype=np.float)
    plt.figure()    
    plt.subplot(1,3,1)
    plt.imshow(positiveMasked)
    plt.subplot(1,3,2)
    plt.imshow(truthMarked)
    plt.subplot(1,3,3)
    composite = 2.*truthMarked + positiveMasked
    plt.imshow(composite)
    #plt.imsave("Test.png",composite)
    positiveScore = truthMarked*positiveMasked
    positiveScore = np.sum(positiveScore)/np.sum(truthMarked)
    print positiveScore

    # negative hits 
    negativeMasked = np.array(negativeHits > 0, dtype=np.float)
    plt.figure()    
    plt.subplot(1,3,1)
    plt.imshow(negativeMasked)
    plt.subplot(1,3,2)
    plt.imshow(truthMarked)
    plt.subplot(1,3,3)
    composite = 2.*truthMarked + negativeMasked
    plt.imshow(composite)
    negativeScore = truthMarked*negativeMasked
    negativeScore = np.sum(negativeScore)/np.sum(truthMarked)
    print negativeScore

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

    fusedPS, bulkNS= Score(fusedPoreResult.stackedHits,bulkPoreResult.stackedHits,"testimages/fusedMarked.png")

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
    
    
    bulkPS, fusedNS = Score(bulkPoreResult.stackedHits,fusedPoreResult.stackedHits,"testimages/bulkMarked.png")   
    
    ## 
    return fusedPS,bulkNS,bulkPS,fusedNS

import pandas as pd

def Assess(
  fusedThreshes = np.linspace(900,1300,3),
  bulkThreshes = np.linspace(900,1300,3),
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
  df.to_hdf(hdf5Name,'table', append=False)
  
  return df,hdf5Name     
  
  



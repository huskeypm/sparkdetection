"""
Generates figure for paper 
"""
import cv2
import numpy as np
import bankDetect as bD
import matplotlib.pylab as plt

class empty():pass
root = "./testimages/"
rawData = root+ 'clahe_Best.jpg'
fuzedData = root+ 'fusedCellTEM.png'
bulkData = root+ 'bulkCellTEM.png'
scale=1.2
fusedThresh=0.2
bulkThresh = 0.46
sigma_n = 1.  # based on Ryan's data 
iters = [5,30]
iters = np.linspace(0,90,10)

def GenFig3(): 
  TestFused()
  TestBulk()

import optimizer
def GenFigROC(loadOnly=False):

  # coarse/fine
  #ft = np.concatenate([np.linspace(0.5,0.7,7),np.linspace(0.7,0.95,15)   ])
  #bt = np.concatenate([np.linspace(0.4,0.55,7),np.linspace(0.55,0.65,15)   ])
  bt = np.linspace(0.05,0.50,10)
  ft = np.linspace(0.05,0.30,10)
  scales = [1.2]  # tried optimizing, but performance seemed to decline quickly far from 1.2 nspace(1.0,1.5,6)  
  hdf5Name = "optimizeinvscale.h5"

  if loadOnly!=True: 
    optimizer.Assess(
        fusedThreshes = ft,
        bulkThreshes = bt,
        scales = scales,
        sigma_n = 1.,
        useFilterInv=True,
        hdf5Name = hdf5Name,
        display=False
      )
  else:
    print "Reading ", hdf5Name 
  import pandas as pd
  df = pd.read_hdf(hdf5Name,'table') 

  optimizer.AnalyzePerformanceData(df,tag='bulk',
                                 normalize=True, roc=True,outName="bulkROC.png")
  optimizer.AnalyzePerformanceData(df,tag='fused',
                                 normalize=True,roc=True,outName="fusedROC.png")



## fused Pore 
def TestFused():
  testCase = empty()
  testCase.label = "fusedEM"
  testCase.name = rawData
  testCase.subsection=[340,440,400,500]
  testCase.outName = "fusedMarkedBest.png"
  #daImg = cv2.imread(testCase.name)
  #daImg = cv2.cvtColor(daImg, cv2.COLOR_BGR2GRAY)
  #raw = daImg[testCase.subsection[0]:testCase.subsection[1],
  #            testCase.subsection[2]:testCase.subsection[3]]
  #imshow(cut)
  DoTest(testCase,fusedThresh=fusedThresh,bulkThresh=bulkThresh,display=True)
  Extrapolate(testCase,tag=testCase.label)

## fused Pore 
def TestBulk():
  testCase = empty()
  testCase.label = "bulkEM"
  testCase.name = rawData
  testCase.subsection=[250,350,50,150] 
  testCase.outName="bulkMarkedBest.png"
  DoTest(testCase,fusedThresh=fusedThresh,bulkThresh=bulkThresh,display=True)
  Extrapolate(testCase,tag=testCase.label)
  
  
def DoTest(testCase,
  fusedThresh=0.2,
  bulkThresh = 0.46,
  display=False
  ):
  fusedPoreResult, bulkPoreResult = bD.TestFilters(
    testCase.name, # testData
    fuzedData,                       # fusedfilter Name
    bulkData,                      # bulkFilter name
    subsection=testCase.subsection, #[200,400,200,500],   # subsection of testData
    fusedThresh = fusedThresh,
    bulkThresh = bulkThresh,
    label = testCase.label,
    sigma_n = sigma_n,
    iters = iters, 
    useFilterInv=True,
    scale=scale,
    display=display,  # to gen images for final copy 
    colorHitsOutName=testCase.outName      
   )        
  testCase.fusedPoreResult = fusedPoreResult
  testCase.bulkPoreResult = bulkPoreResult


def GenFigN():
  testCase = empty()
  testCase.label = "totalEM"
  testCase.name = rawData
  testCase.subsection=None; #[250,350,50,150] 
 
  vals = np.linspace(0.1,0.9,10)
  for i,val in enumerate(vals): 
    testCase.outName="fullBest_%3.1f.png"%val
    DoTest(testCase,fusedThresh=val*fusedThresh,bulkThresh=val*bulkThresh,display=False)
  
import painter
def Extrapolate(
  testCase,
  expPerm = 1.4e-6, # m/s
  bulkPerm = 10**(-6.35),
  fusedPerm= 10**(-6.21),
  tag = None
  ):
  daImg = cv2.imread(testCase.name)
  daImg = cv2.cvtColor(daImg, cv2.COLOR_BGR2GRAY)
  raw = daImg[testCase.subsection[0]:testCase.subsection[1],
              testCase.subsection[2]:testCase.subsection[3]]
    
    
  fusedPoreResult = testCase.fusedPoreResult
  bulkPoreResult = testCase.bulkPoreResult
  fusedPoreResult.labeled = painter.doLabel(fusedPoreResult)
  plt.figure()
  bulkPoreResult.labeled = painter.doLabel(bulkPoreResult)
  ##########
  # create map, assume everthing that isn't characterized is 'unk' 
  # with a permeation somewhere between bulk and fused
  # define your permeation properties
  unkPerm = np.mean([bulkPerm,fusedPerm])

  # if pixel is marked as both fused and bulk, assign pixel as fused
  # inelegant soln, but oh well
  isUnk=0
  isBulk=1
  isFused=2
  permeationIdx  = isUnk * np.ones_like(raw)
  permeationMap  = unkPerm * np.ones_like(raw)
  totArea =  np.prod(np.shape(permeationMap))*1.  
    


  # add bulk 
  isTrue = np.where(bulkPoreResult.labeled)  
  permeationMap[ isTrue  ] = bulkPerm 
  permeationIdx[ isTrue  ] = isBulk  
  #print np.shape( isTrue)
  #areaBulk = np.shape( isTrue )[1]/totArea
  
    
  # add fused
  isTrue = np.where(fusedPoreResult.labeled)  
  permeationMap[ isTrue  ] = fusedPerm
  permeationIdx[ isTrue  ] = isFused  
    
  # get SAs  
  isTrue = np.where(permeationIdx==isBulk)      
  areaBulk  = np.shape( isTrue )[1]/totArea
 
  isTrue = np.where(permeationIdx==isFused)      
  areaFused = np.shape( isTrue )[1]/totArea

  isTrue = np.where(permeationIdx==isUnk)      
  areaUnk = np.shape( isTrue )[1]/totArea

    

  ###############
  plt.subplot(1,2,1)
  plt.imshow(raw,cmap='gray')
  plt.title("Raw")

  fig=plt.subplot(1,2,2)
  fig.set_aspect('equal')
  #plt.pcolormesh(np.flipud(permeationMap),cmap='gray')
  plt.imshow(permeationMap,cmap='gray')
 
  if tag!=None:
    print tag
  print "Frac are: Bulk, Fused,Unk"
  print areaBulk+ areaFused+ areaUnk
  print areaBulk, areaFused, areaUnk

  # compute surface area wighted perm 
  effPerm = areaBulk*bulkPerm + areaFused*fusedPerm + areaUnk*unkPerm
  print "eff perm", effPerm, "log(Peff)", np.log10(effPerm)
  print "exp perm", expPerm
    
  #plt.figure()
  #plt.axes().set_aspect('equal', 'datalim')
  #plt.pcolormesh(np.flipud(permeationMap),shading='gourade',cmap='gray')
  plt.title("Permeation")
  plt.gcf().savefig("extrapolated.png",dpi=300)




  
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
# MAIN routine executed when launching this script from command line 
#
if __name__ == "__maisn__":
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
    if(arg=="-paperfigs"):     
      GenFig3()
    if(arg=="-rocfigs"):     
      GenFig3(loadOnly=True)
  





  raise RuntimeError("Arguments not understood")





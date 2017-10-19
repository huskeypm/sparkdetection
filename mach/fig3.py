import cv2
import numpy as np
import bankDetect as bD
import matplotlib.pylab as plt

class empty():pass
root = "./testimages/"
rawData = root+ 'clahe_Best.jpg'
fuzedData = root+ 'fusedCellTEM.png'
fuzedData = 'workFlowFused.png'           
bulkData = root+ 'bulkCellTEM.png'

fusedThresh=0.2
bulkThresh = 0.46
sigma_n = 1.  # based on Ryan's data 
iters = [5,30]
iters = np.linspace(0,90,10)

## fused Pore 
def TestFused():
  testCase = empty()
  testCase.label = "fusedEM"
  testCase.name = rawData
  testCase.subsection=[340,440,400,500]
  daImg = cv2.imread(testCase.name)
  daImg = cv2.cvtColor(daImg, cv2.COLOR_BGR2GRAY)
  raw = daImg[testCase.subsection[0]:testCase.subsection[1],
              testCase.subsection[2]:testCase.subsection[3]]
  #imshow(cut)
  
  
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
    scale=1.2,
    colorHitsOutName="fusedMarkedBest.png"
   )        


  import painter
  fusedPoreResult.labeled = painter.doLabel(fusedPoreResult)
  plt.figure()
  bulkPoreResult.labeled = painter.doLabel(bulkPoreResult)
  ExtrapToRaw(raw,fusedPoreResult,bulkPoreResult)

def ExtrapToRaw(raw,bulkPoreResult,fusedPoreResult):
  # define your permeation properties
  bulk = 1.
  fused = 5.
  unk = np.mean([bulk,fused])
  
  # create map, assume everthing that isn't characterized is 'unk' 
  # with a permeation somewhere between bulk and fused
  permeationMap  = unk * np.ones_like(raw)
  
  # add bulk 
  permeationMap[ np.where(bulkPoreResult.labeled)  ] = bulk
  print "sestimate surface area"
  
  # add fused
  permeationMap[ np.where(fusedPoreResult.labeled)  ] = fused
  print "estimate seurface area"
  
  plt.subplot(1,2,1)
  plt.imshow(raw,cmap='gray')
  plt.title("Raw")
  
  fig=plt.subplot(1,2,2)
  fig.set_aspect('equal')
  #plt.pcolormesh(np.flipud(permeationMap),cmap='gray')
  plt.imshow(permeationMap,cmap='gray')
  print "mark part that is not assesseed" 
  
  #plt.figure()
  #plt.axes().set_aspect('equal', 'datalim')
  #plt.pcolormesh(np.flipud(permeationMap),shading='gourade',cmap='gray')
  plt.title("Permeation")
  plt.gcf().savefig("extrapolated.png",dpi=300)


  
## fused Pore 
def TestBulk():
  testCase = empty()
  testCase.label = "bulkEM"
  testCase.name = rawData
  testCase.subsection=[250,350,50,150] 
  daImg = cv2.imread(testCase.name)
  cut = daImg[testCase.subsection[0]:testCase.subsection[1],
              testCase.subsection[2]:testCase.subsection[3]]
  #imshow(cut)
  
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
    colorHitsOutName="bulkMarkedBest.png",
    scale=1.2,
    useFilterInv=True,
   )        
  

TestFused()
TestBulk()

import cv2
import bankDetect as bD


# fused Pore 
class empty():pass
root = "./testimages/"
testCase = empty()
testCase.label = "fusedEM"
testCase.name = root+ 'clahe_Best.jpg'
testCase.subsection=[340,440,400,500]
daImg = cv2.imread(testCase.name)
cut = daImg[testCase.subsection[0]:testCase.subsection[1],testCase.subsection[2]:testCase.subsection[3]]
#imshow(cut)

sigma_n = 22. # based on Ryan's data 

if 1: 

 fusedPoreResult, bulkPoreResult = bD.TestFilters(
  testCase.name, # testData
  root+'fusedBase.png',         # fusedfilter Name
  root+'bulkCellTEM.png',        # bulkFilter name
  subsection=testCase.subsection, #[200,400,200,500],   # subsection of testData
  fusedThresh = 1000,#.25,
  bulkThresh = 1050, #.5,
  label = testCase.label,
  sigma_n = sigma_n,
  iters = [0,30],
  display=True
 )        


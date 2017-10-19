import matplotlib.pylab as plt 
import numpy as np 
import cv2
def myplot(img,fileName=None,clim=None):
  plt.axis('equal')
  plt.pcolormesh(img, cmap='gray')
  plt.colorbar()
  if fileName!=None:
    plt.gcf().savefig(fileName,dpi=300)
  if clim!=None:
    plt.clim(clim)

def ReadImg(fileName,renorm=False,bound=False):
    img = cv2.imread(fileName)
    if img is None:
        raise RuntimeError(fileName+" likely doesn't exist")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if bound!=False:
	img=img[bound[0]:bound[1],bound[0]:bound[1]]
    if renorm:# rescaling
    	img = img / np.float(np.amax(img))


    return img  


import scipy.fftpack as fftp

# Prepare matrix of vectorized of FFT'd images
def CalcX(
  imgs,
  debug=False
  ):
  nImg,d1,d2 = np.shape(imgs)
  dd = d1*d2  
  #print nImg, d2
  X = np.zeros([nImg,dd],np.dtype(np.complex128))
    
  for i,img in enumerate(imgs):
    xi = np.array(img,np.dtype(np.complex128))     
    # FFT (don't think I need to shift here)  
    Xi = fftp.fft2( xi )    
    if debug:
      Xi = xi    
    #myplot(np.real(Xi))
    # flatten
    Xif = np.ndarray.flatten(Xi)
    X[i,:]=Xif
  return X  

def TestFilter(
  H, # MACE filter
  I  # test img
):
    #R = fftp.ifftshift(fftp.ifft2(I*conj(H)));
    icH = I * np.conj(H)
    R = fftp.ifftshift ( fftp.ifft2(icH) ) 
    #R = fftp.ifft2(icH) 

    daMax = np.max(np.real(R))
    print "Response %e"%( daMax )
    #myplot(R)
    return R,daMax

# renormalizes images to exist from 0-255
# rescale/renomalize image 
def renorm(img,scale=255):
    img = img-np.min(img)
    img/= np.max(img)
    img*=scale 
    return img

def GetAnnulus(region,sidx,innerMargin,outerMargin=None):
  if outerMargin==None: 
      # other function wasn't really an annulus 
      raise RuntimeError("Antiquated. See GetRegion")

  if innerMargin%2==0 or outerMargin%2==0:
      print "WARNING: should use odd values for margin!" 
  #print "region shape", np.shape(region)

  # grab entire region
  outerRegion,dummy,dummy = GetRegion(region,sidx,outerMargin)
  #print "region shape", np.shape(outerRegion)

  # block out interior to create annulus 
  annulus = np.copy(outerRegion) 
  s = np.shape(annulus)
  aM = outerMargin - innerMargin
  xMin,xMax = 0+aM, s[0]-aM
  yMin,yMax = 0+aM, s[1]-aM
  interior = np.copy(annulus[xMin:xMax,yMin:yMax])
  annulus[xMin:xMax,yMin:yMax]=0. 

  return annulus,interior

def GetRegion(region,sidx,margin):
      subregion = region[(sidx[0]-margin):(sidx[0]+margin+1),
                         (sidx[1]-margin):(sidx[1]+margin+1)]        
      area = np.float(np.prod(np.shape(subregion)))
      intVal = np.sum(subregion)  
      return subregion, intVal, area

def MaskRegion(region,sidx,margin,value=0):
      region[(sidx[0]-margin):(sidx[0]+margin+1),
                         (sidx[1]-margin):(sidx[1]+margin+1)]=value  


import cv2
import numpy as np
from math import *


def ConstructInteriorContour(extContour,
                             distanceParam,
                             imgDim
                             ):

    ## Construction of Interior Contour Given Outer Contour

    # Script to construct an interior contour spaced a predescribed distance away from the exterior contour

    # NOTE: This scales pretty badly and begins to take a lot of time for large images or with contours with many points
    cv2Vers = cv2.__version__


    # Testing to Find Points Within Contour, Calculate Minimum Distance of Each Point Away From Contour, Get Rid of it if Distance is Less than Predescribed Quantity


    validPoints = []
    for x in range(imgDim[1]):
        for y in range(imgDim[0]):
            # positive for interior points
            distance = cv2.pointPolygonTest(extContour, (x,y), True)
            if distance > sqrt(2) / 2 * distanceParam:
                validPoints.append( np.asarray([x,y]))


    # Create Mask of Interior Pixels to Create Interior Polygon


    maskImg = np.zeros(imgDim)
    for point in validPoints:
        maskImg[point[1],point[0]] = 255

    # Apply Gaussian Thresholding and Pull Out Contour


    maskImg = maskImg.astype('uint8')
    blockSize = 3
    thinningParameter = -45
    innerThresh = cv2.adaptiveThreshold(maskImg,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blockSize,thinningParameter)

    if cv2Vers[0] == '3':
        (_,innerContour,_) = cv2.findContours(innerThresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        (innerContour,_) = cv2.findContours(innerThresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    areas = []
    for cnt in (innerContour):
        areas.append(cv2.contourArea(cnt))
    maxArea = np.argmax(areas)
    interiorContour = innerContour[maxArea]

    return interiorContour

def ApplyCLAHE(grayImgList, tileGridSize, clipLimit=2.0, plot=True):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahedimages = []
    for i,img in enumerate(grayImgList):
        clahedImage = clahe.apply(img)
        if plot:
            #plt.figure()
            #imshow(img,cmap='gray')
            #plt.title
            f, (ax1, ax2) = plt.subplots(1,2)
            raw = ax1.imshow(img,cmap='gray')
            f.colorbar(raw, ax=ax1)
            ax1.set_title("Unaltered Image: "+str(i))
            altered = ax2.imshow(clahedImage,cmap='gray')
            f.colorbar(altered, ax=ax2)
            ax2.set_title("CLAHED Image "+str(i))
        clahedimages.append(clahedImage)
    return clahedimages

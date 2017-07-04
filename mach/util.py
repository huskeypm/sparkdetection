import matplotlib.pylab as plt 
import numpy as np 
import cv2
def myplot(img,fileName=None):
  plt.axis('equal')
  plt.pcolormesh(img, cmap='gray')
  plt.colorbar()
  if fileName!=None:
    plt.gcf().savefig(fileName,dpi=300)

def ReadImg(fileName,renorm=False,bound=False):
    img = cv2.imread(fileName)
    if img is None:
        raise RuntimeError(fileName+" likely doesn't exist")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if bound!=False:
	img=img[bound[0]:bound[1],bound[0]:bound[1]]
    if renorm:# rescaling
    	img = img/np.float(np.max(img))


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
    myplot(np.real(Xi))
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
    myplot(R)
    return R,daMax

# renormalizes images to exist from 0-255
def renorm(img):
    img = img-np.min(img)
    img/= np.max(img)
    img*=255
    return img

def GetAnnulus(region,sidx,margin):
      subregion = region[(sidx[0]-margin):(sidx[0]+margin+1),
                         (sidx[1]-margin):(sidx[1]+margin+1)]        
      area = np.float(np.prod(np.shape(subregion)))
      intVal = np.sum(subregion)  
      return subregion, intVal, area


import numpy as np
import Image
import pylab as p
import scipy.fftpack as fftp
from smooth import *

#
# Revision: display imags with correct aspect ratio
#

## IMG 
# rescales image to 0-256 or to predefined values 
def bytscl(d,minVal="none",maxVal="none"):

  if(minVal=="none"):
    minVal = np.min(d)

  g = (d-minVal)


  if(maxVal=="none"):
    maxVal = np.max(g)

  newmax = maxVal-minVal
  g = g/newmax * 256.

  return g

# save grayscale image of 'data' to filename 
def printimg(data,filename):
  w = int((data.shape)[1]/1024. * 16)
  h = int((data.shape)[0]/1024. * 16)
  figprops = dict(figsize=(w,h), dpi=100)
  fig1 = p.figure(**figprops)
  p.subplot(111,aspect='equal')
  p.gray()
  d = np.flipud(data) # for images
  p.pcolormesh(d)
  #p.show()
  p.gcf().savefig(filename)

# short cut for displaying images to screen in ipython
def viewGrayScale(data,flipud=False):
  import Image
  from StringIO import StringIO
 
  p.figure()
  p.subplot(111,aspect='equal')
  if(flipud):
    p.pcolormesh(np.flipud(data))
  else:
    p.pcolormesh(data)

  # saving image to view 
  IO = StringIO()
  p.savefig(IO,format='png')
  IO.seek(0)
 
  Image.open(IO).convert('L').show()

# load images in infiles array
def loadstack(infiles,numChannels=0,keepChannelNum=0):
  n = len(infiles)
  im = Image.open(infiles[0])
  s=im.size[::-1]
  stack = np.zeros((n,s[0],s[1]))

  
  for i,name in enumerate(infiles):
    im = Image.open(name)
    if(numChannels==1):
      di  = np.array(im.getdata()).reshape(im.size[::-1])
    else:
      di = ((np.array(im.getdata()))[:,keepChannelNum]).reshape(s)

    stack[i,:,:] = di


  #rp = np.array([[0,196],[600,796]]) # ref point
  #stack = stack[:,rp[0,0]:rp[0,1], rp[1,0]:rp[1,1]]               


  #printimg(stack[7,:,:],"x.png")

  return stack

## PROC
# Calculate power spectral density as PSD = FFT(n)' FFT(n), where 
# ' designates the complex conjugate 
# assume demeaned
def CalcPsd(data):
  n = data

  from scipy import fft,conj

  fn = fftp.fft2(n)
  psd = np.abs(fn*conj(fn))

  #????
  # psd = fftp.fftshift(psd)

  return psd

# Estimate noise in high frequency region (which is more likely to be
# thermal/white)
# EDIT - could change the region used for noise, but not likely to be
# very important 
def GetHighKNoise(psd):
  size = (np.shape(psd))[0]
  mp = size/2
  marg = size/4
  energy=np.mean(psd[mp-marg:mp+marg,mp-marg:mp+marg])
  return energy 

# Set the minimum value for the PSD (prevents zero-values entries)
def ApplyNoiseFloor(psd,energy):
  psd[psd < energy] = energy

# Deconvolves signal by powerspectral density 
# Substracts datamean from data
# Option is given to apply window to data to prevent edge (Gibbs) effects
def whiten(data, datamean,psd="none",window="none"):
  demean = data-datamean

  if(window!="none"):
    import scipy
    ham = scipy.signal.hamming( (np.shape(demean)[0] ))
    ham = np.outer(ham,ham)
    demean = ham*demean

  if(psd!="none"):
   print "Whitening using input PSD"

  else:
   print  "Computing PSD (with slight smoothing) for whitening "
   psd = CalcPsd(demean)
   psd = smooth(psd,size=2)


  #printimg(np.log(psd),"psd.png")

  # whiten
  unsafeidx = np.where(psd <= 0.0)

  if(len(unsafeidx[0])>0):
    print "There are %d non-positive entries in psd. Avoiding." % len(unsafeidx[0])

  psd[unsafeidx]=1
  w = 1/np.sqrt(psd)
  w[unsafeidx]=0
  M = fftp.fft(demean)
  Mw=M*w
  mw = np.real( fftp.ifft(Mw) );

  return (mw,psd)

# Same as above, but includes demeaning, resizing of psd (if input) and 
# smoothing of psd
# Can support, but need to provide more files to you 
def whiten2(m,psd="none",sigma=3):
  from scipy import ndimage
  from congrid import congrid

  dm = m - np.mean(m)

  fdm = fftp.fft2(dm)

  # create psd
  if(psd=="none"):
    psd = np.abs(fdm*np.conj(fdm))
    psd= ndimage.gaussian_filter(psd, sigma=sigma)

  # check on size
  if(psd.shape!=m.shape):
    print "WARNING: congriding PSD to match data"
    psds = congrid(psd,m.shape,minusone=True)
  else: 
    psds = psd

  # display (see two little specs near 500,500
  #pcolormesh(fftp.fftshift(psd))

  dmw = fftp.ifft2(fdm / psds)
  #pcolormesh(dmw)

  dmw = np.real(dmw) # should be real anyway

  return(dmw,psds)

# Computes PSD based on PSD in each frame 
# Substracts meanImg from images 
def GetAveragePSD(ims,meanImg,size):
  from congrid import congrid
  avgPSD = np.zeros((size,size))
  n= ims.shape[0]
  for i in range(n):
    (wms,psd) = whiten(ims[i,:,:],meanImg)       
    # normalize (Parsevals) 
    psdn= psd/np.prod(np.shape(psd))
    # rescale to size of image
    psdr= congrid(psdn,(size,size),minusone=True)
    avgPSD+=psdr
  
  avgPSD /= np.float(n)
  return avgPSD

# signal to noise ratio
def snr(d):
    m = np.max(d)
    n = np.sqrt(np.var(d))
    # '2' depends on definition of noise (variance or std dev) 
    snr = 2*10*np.log10(m/n)

    print  "Mean: %f Max %f Stddev %f SNR: %f " %  (np.mean(d), m,n,snr)





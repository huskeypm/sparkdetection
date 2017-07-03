"""
Performs basic matched filter test by FFT-correlating a measured signal with an input signal/filter 
"""
# import modules we'll need 
import scipy.fftpack as fftp
import numpy as np
def matchedFilter(
  dimg, # imgs wherein signal (filter) is hypothesized to exist
  daFilter,# signal 
  parsevals=False
  ):
  # placeholder for 'noise' component (will refine later)
  fsC = np.ones(np.shape(dimg))
  
  ## prepare img
  # demean/shift img
  sdimg = fftp.fftshift(dimg - np.mean(dimg))
  # take FFT 
  fsdimg = fftp.fft2(sdimg)

  ## zero-pad filter
  si = np.shape(dimg)
  sf = np.shape(daFilter)
  # add zeros
  zeropad = np.zeros(si)
  zeropad[:sf[0],:sf[1]]=daFilter
  # shift original ligand by its Nyquist
  szeropad = np.roll(\
    np.roll(zeropad,-sf[0]/2+si[0]/2,axis=0),-sf[1]/2+si[1]/2,axis=1)
  f= szeropad

  ## signal
  # shift,demean filter
  sfilter = fftp.fftshift(f- np.mean(f))
  # take FFT
  fsfilter= fftp.fft2(sfilter)

  ## matched filter
  fsh = fsdimg * fsfilter / fsC
  #fsh = np.real( fsh ) 
  sh = fftp.ifft2(fsh)
  h = fftp.ifftshift(sh)
  h = np.real(h)

  ## apply parsevals
  h *= 1/np.float(np.prod(np.shape(h)))
  return h 


########

# find the location and signal-to-noise ratio of the convolutions
def processDetection(corr,threshold):
  maxLoc = np.unravel_index(np.argmax( corr ),sImg) # find location of maximum 
  maxVal = np.max( corr )
  noisevariance = np.var(corr)  # note that we are 'polluting' our estimate of the noise variance with the 
                                                 #signal maximum, so a more careful approach would be to take an 
                                                 # annulus around the maximum
  SNR = maxVal / noisevariance
  if(SNR>threshold):
    print "Detection found at ", maxLoc, "with SNR ", SNR
  else:
    print "No detection" 


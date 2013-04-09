# Function for smooth 2d data

import numpy as np
import scipy.fftpack as fftp
import matplotlib.pylab as p


def shift2d(a,shift):
  ash=np.roll(np.roll(a,shift,axis=0),shift,axis=1)

  return ash        

def smooth1d(a,size=2):
  n = a.shape[0]

  b=np.zeros(n)
  b[0:size]=1.0

  shift = int(size/2.0)
  bsh = np.roll(b,-shift)

  ash = np.roll(a,-shift)
  asmsh=fftp.ifft(fftp.fft(ash)*fftp.fft(bsh))

  asm = np.roll(asmsh,shift)
  asm = np.real(asm)

  return asm
  

def smooth(a,size=2,mode="gaussian"):
  if((a.shape)[0] != (a.shape)[1]):
    print "Not an NxN array)"
    quit()

  n = a.shape[0]

  # define filter
  if(mode=="hat"):
    b=np.zeros((n,n))
    #b[0,0]=b[255,0]=b[0,255]=b[255,255]=1
    b[0:size,0:size]=1.0

    shift = int(size/2.0)
    #bsh=np.roll(np.roll(b,-shift,axis=0),-shift,axis=1)
    bsh = shift2d(b,-shift)

  elif(mode=="gaussian"):
    import scipy.signal as ss
    k = ss.gaussian(n,size) # not exact
    b = np.sqrt(np.outer(k,k))

    shift = int(n/2.0)
    #bsh=np.roll(np.roll(b,-shift,axis=0),-shift,axis=1)
    bsh = shift2d(b,-shift)





  # apply filter
  #ash = np.roll(np.roll(a,-shift,axis=0),-shift,axis=1)
  shift = int(size/2.0)
  ash = shift2d(a,-shift)
  asmsh=fftp.ifft2(fftp.fft2(ash)*fftp.fft2(bsh))

  #asm = np.roll(np.roll(asmsh,shift,axis=0),shift,axis=1)
  asm = shift2d(asmsh,shift)
  asm = np.real(asm)

  return asm


import sys

def doit():
  figprops = dict(figsize=(5,10), dpi=100)
  fig1 = p.figure(**figprops)
  
  n=256
  a = np.random.rand(n*n).reshape((n,n))
  a[ np.where(a>0.5) ] = 1
  a[ np.where(a<=0.5) ] = 0
  #a = np.zeros((n,n))
  #len(np.where(a>0)[0])
  a[156:200,156:200]=1
  a[100:150,100:150]=0
  
  
  
  asm = smooth(a,8)
  
  asm = np.real(asm)
  #asm[ np.where(asm>0.05) ] = 1
  #asm[ np.where(asm<=0.05) ] = 0
  
  
  
  
  p.subplot(2,1,1)
  p.pcolormesh(a)
  p.subplot(2,1,2)
  p.gray()
  p.pcolormesh(asm)
  p.gcf().savefig("x.png")
  #import Image
  #Image.open("x.png").convert('L').show()



if __name__ == "__main__":
  msg="""
Purpose: 
  Smoothing 

Usage:
  script.py <arg>

Notes:
"""
  remap = "none"


  import sys
  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  if(len(sys.argv)==3):
    print "arg"

  for i,arg in enumerate(sys.argv):
    if(arg=="-arg1"):
      arg1=sys.argv[i+1] 




  doit()



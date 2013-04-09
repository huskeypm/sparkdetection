import numpy as np
import matplotlib.pylab as plt
#from lsm_viewer import *
from signaltools import *
#import mach
# sys.path.append('/net/home/huskeypm/bin/Computational_Tools/signalProcessing/')


# whitener seems to eat signal 
whitener=1
marg=20

class empty:pass

# Detection approach (see document) 
# stackSub - raw
# dataDemeaned - avgd data
# frNum - frame number
# sys.path.append('/net/home/huskeypm/bin/Computational_Tools/signalProcessing/')

def ProcessFrame(stackSub,dataDemeaned,dataFiltered,frNum,THRESH_PARM=10000,excerpts=[]):
  ## get frame 
  sfr = stackSub[frNum,:,:]
  fr = dataDemeaned[frNum,:,:]
  sm = dataFiltered[frNum,:,:]
  maxsfr = np.max(stackSub)
  maxfr = np.max(dataDemeaned)
  maxsm  = np.max(dataFiltered)

  
  
  # report
  # don't move
  fvar = np.var(sm)
  fmax = np.max(sm)
  print "Max %f Var %f Max/Var %f" % (fmax,fvar,fmax/fvar) 
  
  
  ## threshold 
  # choose 3 sigma above mean?
  mask = sm > THRESH_PARM
  
  ## find contiguous pixels 
  import scipy.ndimage as ndimage
  label_im, nb_labels = ndimage.label(mask)
  labels = np.unique(label_im)
  
  ## Process each contiguous pixel set
  #N=0 (just background?)
  N=3
  tracksfr = sfr.copy()
  trackfr = fr.copy()
  tracksm = sm.copy()
  nEvents = nb_labels
  for i in np.arange(nb_labels)+1:
    # bracket pixel region 
    slice_x, slice_y = ndimage.find_objects(label_im==i)[0]
    im = sm.copy()
    roi = im[slice_x, slice_y]
    #imshow(roi)
  
    # find max
    peakLoc = np.array(np.unravel_index(np.argmax(roi),np.shape(roi)))
    peakLoc += [slice_x.start,slice_y.start]
    tracksfr[peakLoc[0],peakLoc[1]]=2 * maxsfr
    trackfr[peakLoc[0],peakLoc[1]]=2 * maxfr
    tracksm[peakLoc[0],peakLoc[1]]=2 * maxsm
    #print peakLoc

    # save
    try:
      c =sm[(peakLoc[0]-marg):(peakLoc[0]+marg),(peakLoc[1]-marg):(peakLoc[1]+marg)]
      #print np.shape(excerpts)
    except:
      1
    else:
      if(c.shape[0] ==2*marg and c.shape[1]==2*marg):
        excerpt=empty()
        excerpt.c = c.copy()
        excerpts.append(excerpt)

  # for printing 
  tracksfr[0] = 0; tracksfr[1] = 2*maxsfr
  #print "min ", np.min(tracksfr)
  #print "max ", np.max(tracksfr)
  trackfr[0] = 0; trackfr[1] = 2*maxfr
  tracksm[0] = 0; tracksm[1] = 2*maxsm
  
  
  ## save marked result
  w=4
  plt.gray()
  plt.figure(figsize=[4*w,w])
  plt.subplot(1,3,1)
  plt.pcolormesh(np.flipud(tracksfr))
  plt.subplot(1,3,2)
  plt.pcolormesh(np.flipud(trackfr))
  plt.subplot(1,3,3)
  plt.pcolormesh(np.flipud(tracksm))
  c = plt.gcf()
  n = "processed%.2d" % frNum
  c.savefig(n)
  print "nEvents: %d" % nEvents

  return nEvents
  
  

  
# Main function for calling the algorithm routines 
def doit(fileDir,start=0,numFr=100 ,imgfilter="none",mode="FB"):
  ## params 
  doMovAvg=1
  movAvgForward=1
  movAvgBackward=1
  THRESH_PARM=10000
  
  ## adjust
  s = start+movAvgBackward
  n = numFr-1- movAvgForward
  fproc = np.arange(n)+s

  ## params 
  if(mode=="BL"):
    fRoot = "Feb13-13-113_lsm"
    fExt  = ".tif"
    stackxRange = np.array([[100,175],[100,175]])
    noisexRange = np.array([[400,475],[  0, 75]])

  elif(mode=="FB"):
    # "110510_4_lsm_t0%.3d_c0002.tif"
    fRoot = "110510_4_lsm_t0"
    fExt  = "_c0002.tif"
    #stackxRange = np.array([768,1024])
    stackxRange = np.array([[768,1024],[0,256]])
    #noisexRange = np.array([512,768])
    noisexRange = np.array([[512,768],[0,256]])
    

  
  ## load images
  fread = np.arange(numFr)+start
  infiles = []
  for i in fread:
    infile = fileDir+fRoot+"%.3d" % (i+1)
    infile += fExt
    infiles.append(infile)
  
  stackAll = loadstack(infiles,numChannels=1)
  # 
  
  ## get region I'm interested in 
  #stackSub=stackAll[:,:,768:1024]
  #noiseStack=stackAll[:,:,512:768]
  stackSub=stackAll[:,stackxRange[1,0]:stackxRange[1,1],stackxRange[0,0]:stackxRange[0,1]]
  noiseStack=stackAll[:,noisexRange[1,0]:noisexRange[1,1],noisexRange[0,0]:noisexRange[0,1]]
  #pcolormesh(stackSub[0])
  #pcolormesh(noiseSub[0])


  ## move avg
  if(doMovAvg==1):
    print "Need co-registration here"
    movAvg = np.zeros([fproc.shape[0],stackSub.shape[1],stackSub.shape[2]])
    ctr=0
    for i in fproc:
      movAvg[ctr,:] = np.mean(stackSub[(i-movAvgBackward):(i+movAvgForward),],axis=0)
      ctr+=1
    stackSub = movAvg
    THRESH_PARM=5000
    fproc = np.arange(stackSub.shape[0])

  ## demean images
  avg=np.mean(stackSub,axis=0)
  dataDemeaned = stackSub - avg
  for i in fproc:                  
    d=bytscl(np.flipud(dataDemeaned[i,:,:]))
    plt.pcolormesh(d)
    c=plt.gcf()
    n = "img%.2d" % i
    c.savefig(n)

  ## whiten 
  if(whitener==1):  
    print "Whitening"
    noiseSize = np.shape(noiseStack)[1]
    psd = GetAveragePSD(noiseStack,np.mean(noiseStack,axis=0),noiseSize)
    from congrid import congrid
    psd = congrid(psd,np.shape(dataDemeaned[0,:,:]))

    #pcolormesh(psd)

    margSize = np.shape(avg)[0]
    marg=2
    blank =np.zeros((margSize-2*marg,margSize-2*marg))
    numFr = np.shape(fproc)[0]
    whitened = np.zeros((numFr,blank.shape[0],blank.shape[0]))
    for i in fproc:    
      # this works, whiten2 does not 
      d = dataDemeaned[i,]
      z = whiten(d,np.mean(dataDemeaned,axis=0),psd=psd)[0]
      # trim out edge part, which has edge effects from FFT
      margSize = np.shape(avg)[0]
      marg=2
      z = z[marg:(margSize-marg),marg:(margSize-marg)] 
      #whitened[i,0:z.shape[0],0:z.shape[1]] = z
      whitened[i,:,:]=z

      #
      plt.figure()
      plt.subplot(121,aspect='equal')
      plt.pcolormesh(d)
      plt.title("unwhite")
      plt.subplot(122,aspect='equal')
      plt.pcolormesh(z)
      plt.title("white")
      c=plt.gcf()
      n = "whitimg%.2d" % i
      c.savefig(n)

    THRESH_PARM=7.
    dataDemeaned =whitened

  

  
  ## matched filter (here just a gaussian for now) 
  dataFiltered = np.zeros(np.shape(dataDemeaned))

  # This should be a Gaussian filter, but it looks like I'm using 
  # a boxcar function here. I tried several approaches, so the Gaussian
  # version is elsewhere. 
  if(imgfilter=="none"):
    for i in fproc:                  
      #need to find correct gaussian
      sm = smooth(dataDemeaned[i,:,:],size=4)
      dataFiltered[i,:,:] = sm

  # This is a filter I'm developing, but its not supported currently  
  else: 
    for i in fproc:                  
      #need to find correct gaussian
      print np.shape(imgfilter)
      print np.shape(dataDemeaned[i,:,:])
      sm = mach.matchedfilter(dataDemeaned[i,:,:],imgfilter)
      dataFiltered[i,:,:] = sm
      THRESH_PARM=0.07
  
  ## operate on frame of interest 
  totEvents = 0
  excerpts=[]
  for frNum in fproc:
    nEvents = ProcessFrame(stackSub,dataDemeaned,dataFiltered,frNum,THRESH_PARM=THRESH_PARM,excerpts=excerpts)
    totEvents += nEvents
  
  print "Found %d events" % totEvents

  return dataDemeaned,excerpts

def domach(stack,signals):
  """
  MACH filter tests. Not quite operational 
  """
  n = len(signals)
  sig = signals[0].c
  sigs = np.zeros([n,sig.shape[0],sig.shape[1]])
  for i in range(n):
    #print i 
    #print np.shape(signals[i].c)
    try:
      sigs[i,] = signals[i].c
    except:
     print "Somthin weird"

  #(h,fh)=mach.MACHFilter(sigs,doFFT=False,mode="misc")
  # bad looking filter (h,fh)=mach.MACHFilter(sigs,doFFT=True,mode="misc")
  #(h,fh)=mach.MACHFilter(sigs,doFFT=False,mode="opttradeoff")
  #(h,fh)=mach.MACHFilter(sigs,doFFT=True,mode="opttradeoff")
  #(h,fh)=mach.MACHFilter(sigs,doFFT=False,mode="mach")
  (h,fh)=mach.MACHFilter(sigs,doFFT=True,mode="mach")

  # trim due to edge effects
  h = h[2:38,2:38]

  idx = 10
  r = mach.matchedfilter(stack[idx,],h.copy())

  plt.figure()
  plt.subplot(121,aspect='equal')
  plt.pcolormesh(stack[idx,])
  plt.pcolormesh(h)
  plt.subplot(122,aspect='equal')
  plt.pcolormesh(r)
  print "SNR %f " % (np.max(r)/np.var(r[:,0:100]))

  return h

  
  

import sys
#
# Revisions
#       10.08.10 inception
#

fileDir = "./"

if __name__ == "__main__":
  msg="""
Purpose: 
  Signal detection 

Usage:
  simpleDetect.py -filedir <>

Notes:
  In development

"""
  remap = "none"
  import sys
  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  for i,arg in enumerate(sys.argv):
    if(arg=="-fileDir"):
      fileDir=sys.argv[i+1]
    if(arg=="-whitener"):
      whitener=1

  # Phase 1 - detections using simple kernel 
  numFr=20 # dont recommend more than this for demeaning reasons 
  (stack,signals) = doit(fileDir,numFr=numFr)

  quit()

  # Phase 2 - train signal kernel
  h = domach(stack,signals)
  # Phase 3 - retest using new kernel 
  (stack,signals) = doit(fileDir,numFr=numFr,imgfilter=h)


#In [1]: from simpleDetect import *
#In [2]: (stack,signals) = doit("../110510_4_lsm/",numFr=6)

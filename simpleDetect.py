#figure sys.path.append('/net/home/huskeypm/bin/Computational_Tools/signalProcessing/')
# sys.path.append('/home/huskeypm/sources/sparkdetection/')                    
#doit(fileDir="/Users/huskeypm/localTemp/121201/110510_4_lsm/")
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pylab as plt
#from lsm_viewer import *
from signaltools import *
from simpleDetect import *


#import mach




# whitener seems to eat signal 
whitener=1
margDetect=20
marg=2
altfroot=1
movAvgForward=1
movAvgBackward=1
mode="FB" 

class empty:pass

# Detection approach (see document) 
# stackSub - raw
# dataDemeaned - avgd data
# frNum - frame number
# sys.path.append('/net/home/huskeypm/bin/Computational_Tools/signalProcessing/')

def ProcessDetection(stackSub,dataDemeaned,dataFiltered,frNum,threshParam=10000,
    stackAll=-1,stackxRange=-1,infiles=[],  
    excerpts=[]):

  ## title
  if(len(infiles) > 0): 
    title = infiles[frNum]
  else:
    title ="Fr %d " % frNum

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
  print "Fr %d Max %f Var %f Max/Var %f %s" % (frNum,fmax,fvar,fmax/fvar,title) 
  
  
  ## threshold 
  mask = sm > threshParam
  print threshParam
  
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
  peakLocs=[]
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
    peakLocs.append(peakLoc)
    #print peakLoc

    # save
    try:
      c =sm[(peakLoc[0]-margDetect):(peakLoc[0]+margDetect),(peakLoc[1]-margDetect):(peakLoc[1]+margDetect)]
      #print np.shape(excerpts)
    except:
      1
    else:
      if(c.shape[0] ==2*margDetect and c.shape[1]==2*margDetect):
        excerpt=empty()
        excerpt.c = c.copy()
        excerpts.append(excerpt)

  # for printing 
  tracksfr[0] = 0; tracksfr[1] = 2*maxsfr
  #print "min ", np.min(tracksfr)
  #print "max ", np.max(tracksfr)
  trackfr[0] = 0; trackfr[1] = 2*maxfr
  tracksm[0] = 0; tracksm[1] = 2*maxsm

  #if(stackAll!=-1):
  if 1:
    pls = np.array(peakLocs).reshape(-1, 2)
    

    # indices are flipped,based on how we extracted stackSub from stackAll in the first place
    # (see after loadstack()), so I flip the x/y coordinates from the detection 
    plsOrig = np.array([pls[:,1],pls[:,0]]).T + [marg,marg] + stackxRange[:,0]
    img=dataFiltered[frNum,:,:].copy()
    imgb=stackAll[frNum,:,:].copy()
    maxall = np.max(stackAll)
    maxall = 100 # specifc to one case
    imgb [ imgb > maxall ] = maxall
   
    if(len(peakLocs)>0):
      imgb[ plsOrig[:,1],plsOrig[:,0] ]  = maxall 
      img[ pls[:,0], pls[:,1]]=2*maxsfr
    imgb[0:2,0] = [0,maxall]

    plt.clf()
    plt.subplot(211,aspect="equal")
    plt.pcolormesh(np.flipud(stackAll[frNum,:,:]))
    plt.title(infiles[frNum])
    plt.subplot(212,aspect="equal")
    plt.pcolormesh(np.flipud(imgb))
    plt.pcolormesh(np.flipud(img))
    title = "orig_%.3d.png" % frNum
    g = plt.gcf()
    g.set_size_inches(10.,10.)        
    g.savefig(title)
    1
    
  
  
  ## save marked result
  w=4
  plt.gray()
  #plt.figure(figsize=[4*w,w])
  plt.clf()
  plt.title(title)
  plt.subplot(131,aspect="equal") 
  plt.pcolormesh(np.flipud(tracksfr))
  plt.subplot(132,aspect="equal") 
  plt.pcolormesh(np.flipud(trackfr))
  plt.subplot(133,aspect="equal") 
  plt.pcolormesh(np.flipud(tracksm))
  c = plt.gcf()
  n = "processed%.2d" % frNum
  c.savefig(n)
  print "nEvents: %d" % nEvents

  return nEvents
  

def domovavg(data,start=-1,numFr=-1):
  ## params 
  doMovAvg=1

  if(start==-1):
    start=0;
    numFr = data.shape[0]

  s = start+movAvgBackward
  n = numFr-1- movAvgForward
  fproc = np.arange(n)+s
  
  print "Need co-registration here"
  movAvg = np.zeros([fproc.shape[0],data.shape[1],data.shape[2]])
  ctr=0
  for i in fproc:
    movAvg[ctr,:] = np.mean(data[(i-movAvgBackward):(i+movAvgForward),],axis=0)
    ctr+=1

  return movAvg 

  
# Main function for calling the algorithm routines 
def doit(fileDir,start=0,numFr=100 ,imgfilter="none",mode="FB"):
  ## params 
  doMovAvg=1
  threshParam=10000
  numStdDevs = 3
  

  ## params 
  # Select square subregion for processing and noise estimation  
  if(mode=="BL"):
    fRoot = altfroot     
    fExt  = ".tif"
    start = 43
    numFr = 10

    # hard to see events for this in ordinal image
    stackxRange = np.array([[ 75,150],[100,175]]) # REMEMBER, x/y coords are fliiped in pcolormesh and upsidedown
    stackxRange = np.array([[150,225],[ 90,165]]) # remeber, img is flipped
    noisexRange = np.array([[400,475],[  0, 75]])
    threshParam=10 # obtained by taking max of 'smoothed/filtered' image 
    gaussianFilterSize=4
    determineThreshold=0

  elif(mode=="FB"):
    # "110510_4_lsm_t0%.3d_c0002.tif"
    fRoot = "110510_4_lsm_t0"
    fExt  = "_c0002.tif"
    #stackxRange = np.array([768,1024])
    #stackxRange = np.array([[768,1024],[0,256]])
    stackxRange = np.array([[768,968],[56,256]])
    #noisexRange = np.array([512,768])
    #noisexRange = np.array([[512,768],[0,256]])
    noisexRange = np.array([[ 56,256],[56,256]])
    threshParam=12.          
    gaussianFilterSize=4
    determineThreshold=0

  ## adjust
  s = start+movAvgBackward
  n = numFr-1- movAvgForward
  fproc = np.arange(n)+s

  ## check dim
  dstack = stackxRange[:,1]-stackxRange[:,0]
  dnoise = noisexRange[:,1]-noisexRange[:,0]
  assert(dstack[0]==dstack[1]),"stackxRange must be square"
  assert(dnoise[0]==dnoise[1]),"noisexRange must be square"
  assert(dnoise[0]>=dstack[1]),"noisexRange must be greater or equal to stackxRange for now"
    

  
  ## load images
  fread = np.arange(numFr)+start
  infiles = []
  for i in fread:
    infile = fileDir+fRoot+"%.3d" % (i+1)
    infile += fExt
    infiles.append(infile)
  
  stackAll = loadstack(infiles,numChannels=1)

  # baseline images, so appear on same scale
  stackAll[:,0,0]=np.min(stackAll[:,:,:])
  stackAll[:,0,1]=np.max(stackAll[:,:,:])


  if 0:
    viewGrayScale(stackAll[0,:,:],flipud=True)
  
  ## get region I'm interested in 
  #stackSub=stackAll[:,:,768:1024]
  #noiseStack=stackAll[:,:,512:768]
  stackSub=stackAll[:,stackxRange[1,0]:stackxRange[1,1],stackxRange[0,0]:stackxRange[0,1]]
  noiseStack=stackAll[:,noisexRange[1,0]:noisexRange[1,1],noisexRange[0,0]:noisexRange[0,1]]
  #pcolormesh(stackSub[0])
  #pcolormesh(noiseSub[0])

  # to generate nice image comparison 
  if 0:
    r=np.arange(3)+6   
    r=np.arange(3)+11
    #r = 10
    plt.subplot(311,aspect='equal')
    plt.title("frames: " + p.array2string(r))
    plt.pcolormesh(np.mean(stackAll[r,:,:],axis=0))
    #plt.pcolormesh(stackAll[r,:,:])
    plt.subplot(312,aspect='equal')
    #plt.pcolormesh(stackxRange[0,0]+np.arange(75),stackxRange[1,0]+np.arange(75),stackSub[r,:,:])
    plt.title("stack subplot") 
    plt.pcolormesh(stackxRange[0,0]+np.arange(75),stackxRange[1,0]+np.arange(75),np.mean(stackSub[r,:,:],axis=0))
    plt.subplot(313,aspect='equal')
    plt.title("noise") 
    plt.pcolormesh(noisexRange[0,0]+np.arange(75),noisexRange[1,0]+np.arange(75),np.mean(noiseStack[r,:,:],axis=0))


  ## move avg
  # To diminish some of the incoherent noise 
  if(doMovAvg==1):
    movAvg=domovavg(stackSub)
    stackSub = movAvg

  ## demean images
  # Removes persistent features, like TT network 
  fproc = np.arange(stackSub.shape[0])
  avg=np.mean(stackSub,axis=0)
  avgn=np.mean(noiseStack,axis=0)
  printimg(avg,"average.png")
  dataDemeaned = stackSub - avg
  noiseDemeaned = noiseStack- avgn

  if 0:
    plt.figure()
    for r in fproc:                  
      n = "img_demeaned%.2d" % i
      plt.clf()          
      subplot(131,aspect='equal')
      plt.title("Average") 
      pcolormesh(avg)           
      subplot(132,aspect='equal')
      plt.title("Original") 
      pcolormesh(stackSub[r,:,:])
      subplot(133,aspect='equal')
      plt.title("Demeaned") 
      pcolormesh(dataDemeaned[r,:,:])
      printimg(dataDemeaned[i,:,:],n)




  ## whiten 
  if(whitener==1):  
    print "Whitening"
    noiseSize = np.shape(noiseStack)[1]
    noiseMeanImg = np.mean(noiseStack,axis=0)
    # (wms,psd) = whiten(noiseStack[i,:,:],noiseMeanImg
    psd = GetAveragePSD(noiseStack,noiseMeanImg,noiseSize)                  

    if 0: # for testing psd with white noise 
      # validated (mostly, not quite white, but is probably due to pseudorandom number gen)
      testNoise = np.reshape(np.random.rand(np.prod(noiseStack.shape)),noiseStack.shape)
      testpsd = GetAveragePSD(testNoise,np.mean(testNoise,axis=0),testNoise.shape[1])

    # reshape psd to fit demeaned data 
    printimg(np.log(psd),"logPSD.png")
    from congrid import congrid
    psdn = psd.copy()
    psd = congrid(psd,np.shape(dataDemeaned[0,:,:]))

    #pcolormesh(psd)

    margSize = np.shape(avg)[0]
    blank =np.zeros((margSize-2*marg,margSize-2*marg))
    numFr = np.shape(fproc)[0]
    whitened = np.zeros((numFr,blank.shape[0],blank.shape[0]))
    whitenedn = np.zeros((numFr,blank.shape[0],blank.shape[0]))
    for i in fproc:    
      # this works, whiten2 does not 
      d = dataDemeaned[i,]
      n = noiseDemeaned[i,]
      # actually, data is not mean zero, so mean still needed
      # zeros=np.zeros(d.shape) # already demeaned, so just use zeros
      #z = whiten(d,zeros,psd=psd)[0]
      z = whiten(d,np.mean(d),psd=psd)[0]
      zn = whiten(n,np.mean(n),psd=psdn)[0]
   
      if 0:  # for testing whitening (should go inside whiten function) 
        # Validated 
        testd = np.zeros([75,75])  + np.reshape(np.random.rand(75*75),[75,75])  
        testd[35:39,35:39] = 10
        testpsd = np.zeros([75,75]) + 0.1
        testpsd[:,10:65 ]=1.0  
        subplot(121,aspect='equal')
        pcolormesh( testd )
        subplot(122,aspect='equal')
        pcolormesh( whiten(testd,np.mean(testd),psd=testpsd)[0] )

      # trim out edge part, which has edge effects from FFT
      z = z[marg:(margSize-marg),marg:(margSize-marg)] 
      zn = zn[marg:(margSize-marg),marg:(margSize-marg)] 
      #whitened[i,0:z.shape[0],0:z.shape[1]] = z
      whitened[i,:,:]=z
      whitenedn[i,:,:]=zn

      #
      plt.clf()
      plt.subplot(131,aspect='equal')
      plt.title("unwhite")
      plt.pcolormesh(d)
      plt.subplot(133,aspect='equal')
      plt.title("psd")
      plt.pcolormesh(psd)
      plt.colorbar()
      plt.subplot(132,aspect='equal')
      plt.title("white")
      plt.pcolormesh(z)
      c=plt.gcf()
      n = "whitimg%.2d" % i
      c.savefig(n)

    dataDemeaned =whitened
    noiseDemeaned =whitenedn

  

  
  ## matched filter (here just a gaussian for now) 
  print "Estimating detection threshold from random data (just one slice)" 
  noisevar  = np.var(noiseDemeaned) # data has been whitened and demeaned)
  i = 0 #
  false=noiseDemeaned[i,:,:]
  #false = np.reshape(noiseVar*np.random.rand(np.prod(false.shape)),false.shape)
  #false -= np.mean(false)
  sm = smooth(false,size=gaussianFilterSize)
  if determineThreshold:
    threshParam = np.max(sm) + numStdDevs * np.var(sm) 
    

  # Detection using Gaussian filter, but it looks like I'm using 
  # a boxcar function here. I tried several approaches, so the Gaussian
  # version is elsewhere. 
  dataFiltered = np.zeros(np.shape(dataDemeaned))
  if(imgfilter=="none"):
    for i in fproc:                  
      #need to find correct gaussian
      sm = smooth(dataDemeaned[i,:,:],size=gaussianFilterSize)
      dataFiltered[i,:,:] = sm

      if 0:
        plt.clf()
        plt.subplot(121,aspect='equal')
        pcolormesh(dataDemeaned[i,:,:])
        colorbar()
        plt.subplot(122,aspect='equal')
        pcolormesh(dataFiltered[i,:,:])    
        colorbar()


  # This is a filter I'm developing, but its not supported currently  
  else: 
    for i in fproc:                  
      #need to find correct gaussian
      print np.shape(imgfilter)
      print np.shape(dataDemeaned[i,:,:])
      sm = mach.matchedfilter(dataDemeaned[i,:,:],imgfilter)
      dataFiltered[i,:,:] = sm
      threshParam=0.07
  
  ## operate on frame of interest 
  totEvents = 0
  excerpts=[]
  for frNum in fproc:
    nEvents = ProcessDetection(
      stackSub,dataDemeaned,dataFiltered,frNum,threshParam=threshParam,excerpts=excerpts,
      stackAll=stackAll,stackxRange=stackxRange,infiles=infiles)
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

  REMEMBER TO ADD LSM READONER FROM JUSTIN 

"""
  remap = "none"
  import sys
  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  numFr=20 # dont recommend more than this for demeaning reasons 
  for i,arg in enumerate(sys.argv):
    if(arg=="-fileDir"):
      fileDir=sys.argv[i+1]
    if(arg=="-whitener"):
      whitener=1
    if(arg=="-froot"):
      altfroot=sys.argv[i+1]
    if(arg=="-mode"):
      mode= sys.argv[i+1]
    if(arg=="-altfroot"): 
      altfroot= sys.argv[i+1]


  # Phase 1 - detections using simple kernel 
  print "Using -fileDir %s" % fileDir
  (stack,signals) = doit(fileDir,numFr=numFr,mode=mode)  

  quit()

  # Phase 2 - train signal kernel
  h = domach(stack,signals)
  # Phase 3 - retest using new kernel 
  (stack,signals) = doit(fileDir,numFr=numFr,imgfilter=h)


#In [1]: from simpleDetect import *
#In [2]: (stack,signals) = doit("../110510_4_lsm/",numFr=6)

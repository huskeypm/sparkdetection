from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np 
from matplotlib.figure import Figure
import hexagon
import util 
# has annoying padding, but screw it 
def GenerateImageFromScatter(scatterData,nxPix=(512),size=(10)):
    # if we want nxPix in the x and y directions, we need to create a figure with size w = nxPix/dpi
    dpi =160 # apparently
    dim = nxPix/dpi
    
    # call commands for generating saveable figure 
    fig = Figure([dim,dim],dpi=dpi) 
    #fig = Figure()                  
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    print "max", np.max(scatterData[:,0]) , np.max(scatterData[:,1])
    print "min", np.min(scatterData[:,0]) , np.min(scatterData[:,1])
#    print "scatData1", np.shape(scatterData) #[:,0], "scatData2", scatterData[:,1]
    sz = (11.7/(np.max(scatterData[:,1]) - np.min(scatterData[:,1])))**1.4*size
    print "sz ", sz

    size = sz*(12.5/(np.max(scatterData[:,0]) - np.min(scatterData[:,0])))**1.4
    print "size ", size 
    # do p[lotting]
    #ax.autoscale(tight=True)
    ax.set_aspect('equal')
    ax.scatter(scatterData[:,0],scatterData[:,1],s=size,c='k')#, marker='o')
    ax.axis('off')
    
    
    # store plot to strong 
    canvas.draw()       # draw the canvas, cache the renderer
    data = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data[:,:,2]
    return data

def GenLattice(mode="perfect", angle=0):
  # generate lattice   
  if mode=="perfect":
        pts = hexagon.drawHexagonalGrid(8,8,size=1)
        data = GenerateImageFromScatter(pts, 512, 10)

  elif mode == "twinned":
        pts = hexagon.drawTwinnedHexagonalGrid(8,16,size=1) 
        data = GenerateImageFromScatter(pts, 512, 10 )
  elif mode == "multi":

	pts = hexagon.drawMultiTwinnedHexagonalGrid(8,8,size=1)    
        data = GenerateImageFromScatter(pts, 512, 10)
  else:
	pts = hexagon.drawReflectionHexagonalGrid(angle,8,8,size=1)
        data = GenerateImageFromScatter(pts, 512, 10)
    #plt.axis('equal')
    #plt.scatter(perfectPts[:,0],perfectPts[:,1],100)

  ## convert to image
  #data = GenerateImageFromScatter(pts)
  data = np.array(data,dtype=float)  
  # white pts on blk bg
  data = np.max(data) - data
  # normalize/rescale
  #data = data - np.min(data)
  #data /= np.max(data)  
  # PKH 
  rscl = util.renorm(data,scale=1.)
    
  return  rscl    


import cv2
# blurs and adds poisson noise 
def AddRealism(img,noiseFloor=False, scale = 0.3,Gauss =True): 
    # blur
    if Gauss:
      blur = cv2.GaussianBlur(np.array(img,dtype=float),(3,3),0)
      dim = np.shape(blur)
      blur = blur - np.min(blur)
      blur /= np.max(blur)
           
    # add noise 
    #noise = np.random.randn( np.prod(dim))
    # this should be 0.3, noise cranked for ROC
    mean = 0.3
    noise = scale*np.random.poisson(mean, np.prod(dim))
    thresh = 1.
    noise[ np.where(noise>thresh)]=thresh
    noise = np.reshape(noise,dim)
    realImg = blur + noise

    if noiseFloor:
      scale=0.4
      dim = np.shape(realImg)
      noise = scale*np.random.randn(np.prod(dim))
      noise = np.reshape(noise,dim)
      realImg = realImg+ noise




    return realImg
    


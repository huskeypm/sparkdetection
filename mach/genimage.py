from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np 
from matplotlib.figure import Figure
import hexagonFFF
# has annoying padding, but screw it 
def GenerateImageFromScatter(scatterData,nxPix=(1024),size=(10)):
    # if we want nxPix in the x and y directions, we need to create a figure with size w = nxPix/dpi
    dpi =160 # apparently
    dim = nxPix/dpi
    
    # call commands for generating saveable figure 
    fig = Figure([dim,dim],dpi=dpi) 
    #fig = Figure()                  
    canvas = FigureCanvas(fig)
    ax = fig.gca()
   
    #print "scatData1", scatterData[:,0], "scatData2", scatterData[:,1]

 
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
        pts = hexagonFFF.drawHexagonalGrid(16,16,size=2)
  elif mode == "twinned":

        pts = hexagonFFF.drawTwinnedHexagonalGrid(16,32,size=2) 
  elif mode == "multi":
	pts = hexagonFFF.drawMultiTwinnedHexagonalGrid(16,16,size=2)    
  else:
	pts = hexagonFFF.drawReflectionHexagonalGrid(angle,16,16,size=2)
    #plt.axis('equal')
    #plt.scatter(perfectPts[:,0],perfectPts[:,1],100)
  print np.shape(pts)

  ## convert to image
  data = GenerateImageFromScatter(pts)
  data = np.array(data,dtype=float)  
  # white pts on blk bg
  data = np.max(data) - data
  # normalize/rescale
  data = data - np.min(data)
  data /= np.max(data)  
  print np.shape(data)  
    
  return  data    

import cv2
# blurs and adds poisson noise 
def AddRealism(img,noiseFloor=False): 
    # blur
    blur = cv2.GaussianBlur(np.array(img,dtype=float),(3,3),0)
    dim = np.shape(blur)
    blur = blur - np.min(blur)
    blur /= np.max(blur)

    # add noise 
    #noise = np.random.randn( np.prod(dim))
    scale =0.3
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
    


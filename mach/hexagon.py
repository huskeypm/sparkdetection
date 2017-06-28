"""
Generates hexagonal close packed vertices
"""
import numpy as np 
class center:
    def __init__(self,x,y):
        self.x=x
        self.y = y

def hex_corner(center, size=1, i=0):
    angle_deg = 60. * i   + 30.
    angle_rad = np.pi / 180. * angle_deg
    return [center.x + size * np.cos(angle_rad),
                 center.y + size * np.sin(angle_rad)]



def drawHexagon(origin=center(0,0),i=0,j=0,size=1, wcenter=True):
    # should go elsewhere
    coords = []
    height = size * 2    
    vertDist = height*3/4.
    width = np.sqrt(3)/2. * height
    horzDist = width

    if j%2==1:
      xoffset = -np.sqrt(3)/2 * height/2.
    else:     
      xoffset = 0  
    yoffset = size*np.sin(np.pi/6.) 

        
    hexCtr = center(origin.x+i*width+xoffset, origin.y+j*(size+yoffset))

    # center
    if wcenter:
      coords.append([hexCtr.x,hexCtr.y])

    # edges 
    for i in range(6):
      coord = hex_corner(hexCtr,size=size,i=i)
      coords.append(coord)      

    coords = np.array(coords)#,[7,2]
    return coords 

# Draws perfect hexagonal grid 
def drawHexagonalGrid(
  xIter, # unit cells in x direction
  yIter,
  size=1,# size of unit cell
  edged=True  # return entries for which x,y > 0,0
  ):
    
  # generate 
  origin = center(0.,0.)
  coords = []        
  for i in range(xIter): 
    for j in range(yIter): 
        coord = drawHexagon(origin,i,j,size)
        coords.append(coord)

  # reorg into nparray 
  dim = np.shape(coords)
  coords = np.reshape(coords,[np.prod(dim)/2.,2])

  # return only those entries w x,y > 0,0 
  if edged:
    coords= [ x for x in np.vsplit( coords,coords.shape[0] )  if x[0,0]>=0. and x[0,1]>=0.]
    dim = np.shape(coords)
    coords = np.reshape(coords,[np.prod(dim)/2.,2])
 
  return coords 

# Draws a hexagonal grid with two twinned regions; interface (reflection) is at x=0 
def drawTwinnedHexagonalGrid(
  xIter,
  yIter,
  size=1): 

  coords = drawHexagonalGrid(xIter/2,yIter/2,size=size)
  coords2 = np.copy(coords)
  # reflect
  coords2[:,0]*=-1
  # shift by 1 bond distance
  coords2[:,0]-= size 
  final = np.concatenate((coords,coords2),axis=0)
  # moves reflection plane to x=0
  final[:,0]+= 0.5*size 

  return final 
  
  
 




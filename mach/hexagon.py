"""
Generates hexagonal close packed vertices
"""
import numpy as np 
class center:
    def __init__(self,x,y):
        self.x=x
        self.y = y
class twins:
  def __init__(self,x1,x2,y1,y2):
	self.x1 = x1
	self.x2 = x2
	self.y1 = y1
	self.y2 = y2
	
def det(a,b):
	return a[0]*b[1] - a[1]*b[0]
def knowTwins(listTwins, coords):
  modifiedList = listTwins
  listLength = len(listTwins)
  important = (listLength**2 - listLength)/2
  for i,instance1 in enumerate(listTwins):
    if i == listLength-1:
      break
    for j,instance2 in enumerate(listTwins[i:,]):
      xdiff = (paired1.x1 -paired1.x2, paired2.x1 - paired2.x2)      
      ydiff = (paired1.y1 -paired1.y2, paired2.y1 - paired2.y2)      
      div = det(xdiff,ydiff)
      m1 = (paired1.y1 -paired1.y2)/(paired1.x1 -paired1.x2)
      m2 = (paired2.y1 -paired2.y2)/(paired2.x1 -paired2.x2)
      d = (det(paired1),det(paired2))
      x = det(d, xdiff)/div
      y = det(d, ydiff)/div
      if x > np.max(coords[:,0]) or x > np.min(coords[:,0]) or x<np.min(coords[0,:]) or y< np.min(coords[0,:]):
	print instance1, " ignored"
      else:
	if ((paired1.x1 < x) and (paired1.x2 >x)) or ((paired1.x1 > x) and (paired1.x2 <x)):
		modifiedList[instance1] = (x,paired1.x1, y, paired1.y1)
 	else: 
		modifiedList[instance2] = (x,paired2.x2,y,paired2.y2) # this is dumb and probably doesnt work, will finish later

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
  edged=True,  # return entries for which x,y > 0,0
  cent = [0,0]
  ):
  origin = center(cent[0],cent[1])  
  coords = []        
  for i in range(xIter): 
    for j in range(yIter): 
        coord = drawHexagon(origin,i,j,size)
        coords.append(coord)

  dim = np.shape(coords)
  coords = np.reshape(coords,[np.prod(dim)/2.,2])
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
  final[:,0]+= 0.5*size
  return final 

#def remover(arr):
#  copy = np.zeros_like(arr)  
  
def drawMultiTwinnedHexagonalGrid(
  xIter,
  yIter,
  size=1): 

  coords = drawHexagonalGrid(xIter,yIter,size=size)
  final = coords

  #grabbing the x values for overlap
  less = np.where(5<final[:,0])
  Less = less[0]
  more = np.where(final[:,0]<6)
  More = more[0]
  
  #array-ifying
  this1 = np.asarray(More)
  this2 = np.asarray(Less)
  
  #pulling the overlapping region out to "skip" portion of unit cell to mimick twinning
  intersect = np.intersect1d(this1,this2)
  temp = np.delete(final,intersect, axis = 0)
  newL= np.where(temp[:,0]<6)[0]
  newR= np.where(temp[:,0]>4)[0]

  #grabbing the newpoints to reform image
  this3 = np.asarray(newL)
  this5 = np.asarray(newR)
  leftInt = temp[0:(len(this3)),:]
  start = (np.max(this3))
  rightInt = temp[(start):,:]
 
  #trouble starts here, whenever right+= is used a duplicate of the first point in rightInt appears
  leftInt*=10
  rightInt*=10 

  L = leftInt.tolist()
  R = rightInt.tolist()
  Ls = [tuple(i) for i in L]
  Rs = [tuple(i) for i in R]
  newLeft = set(Ls)
  newRight = set(Rs)

  newL = np.array(list(newLeft))
  newR = np.array(list(newRight))
  newLeft =np.around(newL, decimals=1)
  newRight = np.around(newR, decimals=1)
  maxer = np.max(newLeft[:,0])
  miner = np.min(newRight[:,0])
  nopeL = np.where(newLeft[:,0]== maxer)[0]
  nopeR = np.where(newRight[:,0] <60)[0]
  #print " Righty ", newRight[nopeR,0]  
  Lefty = np.delete(newLeft, nopeL, axis =0)
  Righty = np.delete(newRight, nopeR, axis=0)
  newLeft = Lefty
  newRight = Righty
  #print miner, " nopeR"  
  #print "newRight ", newRight
  newLeft[:,0] +=3.5*size
  newRight[:,0]-=3.5*size
  finals = np.concatenate((newLeft,newRight),axis=0)
  finals/=10
  return finals

def drawReflectionHexagonalGrid(
  angle,
  xIter,
  yIter,
  size=1,
  ):
  coords = drawMultiTwinnedHexagonalGrid(xIter, yIter, size)
  newCoords = randrot(coords, angle)
#  maxX = np.max(coords[:,0])
#  minX = np.min(coords[:,0])
#  avgX = (maxX+minX)/2
#  maxY = np.max(coords[:,1])
#  minY = np.min(coords[:,1])
#  avgY = (maxY+minY)/2
#  newCoords[:,0] -=avgX
#  newCoords[:,1] -=avgY 
#  newCoords[:,0] *= np.cos(angle)
#  newCoords[:,1] *= np.sin(angle)
  return newCoords 

def rot(t):
    R = np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    return R

def randrot(coords, angle = 0):
    if angle != 0:
      R = rot( angle )
      c = np.dot(R,np.transpose(coords))
      c = np.transpose(c)

    else:
      t = (np.random.rand(1) * np.pi/4.)[0]
      R = rot( t )
      c = np.dot(R,np.transpose(coords))
      c = np.transpose(c)
    return c 


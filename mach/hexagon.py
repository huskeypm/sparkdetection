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
	
  #def line(self, x1,
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
  # generate 
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
  print np.shape(coords) 
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
  #myarr = np.ones(200) 
  #for vals in range(200):
  #  myarr[vals] = 694+vals
  #temp = np.delete(final,myarr, axis = 0)
  #print np.shape(temp)
  #print temp[301]

  return final 
  
  
def drawMultiTwinnedHexagonalGrid(
  xIter,
  yIter,
  size=1): 

  coords = drawHexagonalGrid(xIter,yIter,size=size)
  #coords2 = np.copy(coords)
  # reflect
  #coords2[:,0]*=-1

  # shift by 1 bond distance
  #coords2[:,0]-= size 

  #final = np.concatenate((coords,coords2),axis=0)
  # moves reflection plane to x=0
  #final[:,0]+= 0.5*size 
  
  # OK, so the plan is to send a call to the fxn knowTwins
  # Once we find where the twins are, we strike throug the 
  final = coords
  #myarr = np.ones(200) 
  #for vals in range(200):
  #  myarr[vals] = 700+vals
  less = np.where(20<final[:,0])


  #my = less[:,1]



  #print "less", less
  Less = less[0]
  #Less = less.reshape(len(less),1)
  print "shape less", np.shape(Less)
  more = np.where(final[:,0]<22)
  More = more[0]
  print "more", np.shape(more[0][:])
  

  this1 = np.asarray(More)
  this2 = np.asarray(Less)


  intersect = np.intersect1d(this1,this2)
  print "intersect", intersect
  #toDelete = np.where(interect)
  temp = np.delete(final,intersect, axis = 0)
  print np.shape(temp)
  print temp[301]

  #mylist = knowTwins[]
  return temp 
  
  

def drawReflectionHexagonalGrid(
  angle,
  xIter,
  yIter,
  size=1,
  ):
  
  coords = drawHexagonalGrid(xIter/2,yIter/2,size=size)
  coords2 = np.copy(coords)
  newCoords = np.copy(coords)
  newCoords[:,0] *= np.cos(angle)
  newCoords[:,1] *= np.sin(angle)
  coords2[:,0] *= -1*np.cos(angle)
  coords2[:,1] *= -1*np.sin(angle)
 


  print coords2[20,0]

 
  # reflect
  #coords2[:,0]*=-1*np.cos(angle)
  
  # shift by 1 bond distance
  #coords2[:,0]-= size*np.cos(angle)

  final = np.concatenate((newCoords,coords2),axis=0)
  # moves reflection plane to x=0
  final[:,0]+= 0.5*size
  #myarr = np.ones(200) 
  #for vals in range(200):
  #  myarr[vals] = 694+vals
  #temp = np.delete(final,myarr, axis = 0)
  #print np.shape(temp)
  #print temp[301]

  return final


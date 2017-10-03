import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy.misc import toimage
import imutils
import imtools as it
import painter as Paint
import random as rand



class AFusedBox():   # format is (y,x)    is (36,18) for bulk    is (26,44) for fused
  def __init__(self,y,x): #x1, y1 is top left corner
    self.x1 = x - 22
    self.x2 = x + 22
    self.y1 = y - 13
    self.y2 = y + 13
  #def inside(self,x1,y1)
def buildBox(dims, filterType = 'fused'):
  if filterType == 'fused':
        mybox = AFusedBox(dims[0],dims[1])
  elif filterType == 'bulk':
        mybox = ABulkBox(dims[0],dims[1])
  return mybox

    
    
class ABulkBox():   # format is (y,x)    is (36,18) for bulk    is (26,44) for fused
  def __init__(self,y,x): #x1, y1 is top left corner
    self.x1 = x - 9
    self.x2 = x + 9
    self.y1 = y - 18
    self.y2 = y + 18




#k means clustering to grab centers of boxes

def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

def find_centers(X, K):
    # Initialize to K random centers
    oldmu = rand.sample(X, K)
    mu = rand.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)

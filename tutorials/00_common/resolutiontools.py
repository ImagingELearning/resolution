import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import tifffile as tiff
import matplotlib.patches as mpatches
import skimage.filters as flt

def COG(gimg1,begin,end) :
    [x,y]=np.meshgrid(np.linspace(0,gimg1.shape[1]-1,gimg1.shape[1]),
                  np.linspace(0,gimg1.shape[0]-1,gimg1.shape[0]))

    pimg = gimg1
    th=(pimg.max()-pimg.min())*0.3+pimg.min()
    pimg[pimg<th]=0
    cog = (x[:,begin:end]*pimg[:,begin:end]).sum(axis=1)/pimg[:,begin:end].sum(axis=1)
    pos = np.arange(len(cog))
    return cog,pos

def pointDistance(coefs,r,c) :
    d=-(r*coefs[0]-c+coefs[1])/np.sqrt(coefs[0]**2+1.0)
    
    return d

def computeDistanceField(coefs,size) :
    r,c=np.meshgrid(np.arange(0,size[1]), np.arange(0,size[0]))
    
    d=-(c*coefs[0]-r+coefs[1])/np.sqrt(coefs[0]**2+1.0)
    
    return d



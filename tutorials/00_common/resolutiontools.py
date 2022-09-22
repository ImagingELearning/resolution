import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import tifffile as tiff
import matplotlib.patches as mpatches
import skimage.filters as flt
from scipy.special import wofz


def COG(img,begin,end) :
    [x,y]=np.meshgrid(np.linspace(0,img.shape[1]-1,img.shape[1]),
                  np.linspace(0,img.shape[0]-1,img.shape[0]))

    pimg = img
    th=(pimg.max()-pimg.min())*0.3+pimg.min()
    pimg[pimg<th]=0
    cog = np.array(x[:,begin:end]*pimg[:,begin:end]).sum(axis=1)/pimg[:,begin:end].sum(axis=1)
    pos = np.arange(len(cog))
    return cog,pos

def fitEdgeLine(cog,N) :
    pos  = np.arange(len(cog))
    idx  = np.argsort(cog) # indirect sorting, returns a vector with indices of the sorting 
    line = np.polyfit(pos[idx[N:-N]],cog[idx[N:-N]],1) # exclude N most deviating points at beginning and end 
    
    return line 

def pointDistance(coefs,r,c) :
    d=-(r*coefs[0]-c+coefs[1])/np.sqrt(coefs[0]**2+1.0)
    
    return d

def computeDistanceField(coefs,size) :
    r,c=np.meshgrid(np.arange(0,size[1]), np.arange(0,size[0]))
    
    d=-(c*coefs[0]-r+coefs[1])/np.sqrt(coefs[0]**2+1.0)
    
    return d

def edgeProfile(x,decimals = 2) :
    cog,_  = COG(flt.sobel(x),0,x.shape[1]-1)
    
    line = fitEdgeLine(cog,5)
    dist = computeDistanceField(line,x.shape)
    dist = np.around(dist,decimals=decimals)

    ud   = np.unique(dist)
    
    profile = np.zeros(ud.shape)
                       
    for idx,d in enumerate(ud) :
        profile[idx]=x[dist==d].mean()
        
        
    return profile,ud

def G(x, alpha):
    """ Return Gaussian line shape at x with HWHM alpha """
    return np.sqrt(np.log(2) / np.pi) / alpha\
                             * np.exp(-(x / alpha)**2 * np.log(2))
def G2(x,alpha, x0, A) :
    return A*G(x-x0,alpha)

def V(x, alpha, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = alpha / np.sqrt(2 * np.log(2))

    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi)

def V2(x,alpha,gamma,x0,A) :
    return A*V(x-x0,alpha,gamma)

def L(x, gamma):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return gamma / np.pi / (x**2 + gamma**2)

def FWHM_Gauss(sigma) :
    return 2*sigma*np.sqrt(2*np.log(2))

def FWHM_Lorentz(gamma) :
    return 2*gamma

def FWHM_Voigt(coefs) :
    return 0.5346 * FWHM_Lorentz(coefs[1]) \
        + np.sqrt( 0.2166*FWHM_Lorentz(coefs[1])**2+FWHM_Gauss(coefs[0])**2)
import numpy as np
import scipy.fftpack as fftp
import util 
import matplotlib.pylab as plt 
# [imgRows imgCols timeSamples] = size(volumes{1});
def MACH2(
    images,
    alpha = 50,  
    beta = 1e-12, 
    gamma = 1e-12 
    ):
    ## Create rasterized (vectorized ) FFT images 
    N, imgRows, imgCols = np.shape( images )
    #d = imgRows * imgCols * timeSamples;
    d = imgRows * imgCols
    #x = zeros(d, N);
    x = np.zeros([d,N],np.dtype(np.complex128))
    #for i = 1 : N
    doFFT=True
    for i in range(N):
        #fft_volume = fft3(double(volumes{i}));
        #x(:,i) = fft_volume(:);
        if doFFT:
          fft_volume = fftp.fftn(images[i,:,:])
        else:
          fft_volume = images[i,:,:]  
        #print np.shape(fft_volume)
        x[:,i] = np.ndarray.flatten( fft_volume )
    #end
    #mx = mean(x, 2);
    #c = ones(d,1);
    #dx = mean(conj(x) .* x, 2);
    #temp = x - repmat(mx, 1, N);   
    #sx = mean(conj(temp) .* temp, 2);
    mx = np.mean(x,axis=1)
    #util.myplot( fftp.ifftn(np.reshape(x[:,i],[imgRows, imgCols])))
    #util.myplot( fftp.ifftn(np.reshape(mx,[imgRows, imgCols])))
    #util.myplot( np.reshape(mx,[imgRows, imgCols]))
    #print np.shape(mx)

    ## C
    c = np.ones(d)

    ## Dx
    dx = np.mean(np.conj(x) * x, axis=1);
    #print np.shape(dx)

    ## Sx
    diff = np.transpose( x.transpose()-mx.transpose() )
    sx = np.mean( np.conj(diff)*diff,axis=1)
    #util.myplot( np.reshape(diff[:,2],[imgRows, imgCols]))
    #plt.colorbar()

    ## Define denominator 
    
    ## Calc filter 
    #h_den = (alpha * c) + (beta * dx) + (gamma * sx);
    h_den = (alpha * c) + (beta * dx) + (gamma * sx);
    #print np.shape(h_den)

    #h = mx ./ h_den;
    #h = reshape(h, [imgRows, imgCols, timeSamples]);
    #h = real(ifft3(h)); 
    #h = uint8(scale(h, min3(h), max3(h), 0, 255)); 
    h = mx / h_den;
    h = np.reshape(h, [imgRows, imgCols]);
    h = np.real(fftp.ifftn(h)); 
    util.myplot(h)
    h = util.renorm(h)
    
    return h


    


## they do FFT sequentially. Not clear why
def fftcorr(I,T,correlationThreshold=1.6e8):
    fI = fftp.fftn(I)
    fT = fftp.fftn(T)
    c = np.conj(fI)*fT
    corr = fftp.ifftn(c)
    corr = np.real(corr)
    
    whereMax,result= isHit(corr,daMaxThresh = correlationThreshold)
    return corr, whereMax,result 

## checks for match in data based on peak/sidelobe thresholds
def isHit(corr,
    peakMargin=3,
    lobeMargin=20,
    daMaxThresh=1.6e8,
    sideLobeThresh=1.03):

    dummyMargin = lobeMargin - 10
    daMax = np.max(corr)
    whereMax = np.unravel_index(np.argmax(corr),np.shape(corr))
    
    # amplitude of max is generally not enough; folks commonly use
    # sidelobe ratio (need to review this)
    # here's a cheap alterative
    #plt.figure()
    # shift so easier to 'bracket' peak
    scorr = fftp.fftshift(corr)
    #rescale = np.float( 255 * np.prod( np.shape(scorr )) )
    #scorr /= rescale
    sidx= np.argmax(scorr)
    sidx = np.unravel_index(sidx,np.shape(scorr))
    #util.myplot(scorr)
    plt.plot(scorr[:,sidx[1]])
    print sidx
    
    # peak region 
    peak, peakInt,peakArea = util.GetAnnulus(scorr,sidx,peakMargin)
    peakNorm = peakInt/peakArea
    
    # side loberegion 
    dummy, dummyInt,dummyArea = util.GetAnnulus(scorr,sidx, dummyMargin)
    lobe, lobeInt,lobeArea = util.GetAnnulus(scorr,sidx,lobeMargin)
    lobeInt -= dummyInt
    lobeArea-= dummyArea
    lobeNorm = lobeInt/lobeArea
    peakToLobe = peakNorm/lobeNorm
    print "%e, %e, %f"%(peakNorm,lobeNorm, peakToLobe)    
    
    if peakNorm > daMaxThresh and peakToLobe > sideLobeThresh:
      print "POSITIVE"
      result = True  
      
    else:
      print "NEGATIVE"
      result = False
    return whereMax,result 
    





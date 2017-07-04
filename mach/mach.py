# [imgRows imgCols timeSamples] = size(volumes{1});
def MACH2(images):
    ## Create rasterized (vectorized ) FFT images 
    N, imgRows, imgCols = np.shape( images )
    #d = imgRows * imgCols * timeSamples;
    d = imgRows * imgCols
    #x = zeros(d, N);
    x = np.zeros([d,N],np.dtype(complex128))
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

    ## Sx
    diff = np.transpose( x.transpose()-mx.transpose() )
    sx = np.mean( np.conj(diff)*diff,axis=1)
    #util.myplot( np.reshape(diff[:,2],[imgRows, imgCols]))
    #plt.colorbar()

    ## Define denominator 
    alpha = 50;  
    beta = 1e-12; 
    gamma = 1e-12;
    
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
    h = renorm(h)
    plt.colorbar()
    
    return h


    


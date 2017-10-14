#### fused EM
IMG=fusedEM
TYPE=fused
convert ${IMG}_${TYPE}_0.png -geometry 640x A.png
convert ${IMG}_${TYPE}_30.png -geometry 640x B.png
convert ${IMG}.png -geometry 640x C.png

  montage         A.png \
                  B.png \
                  C.png \
          -geometry '+0+0+0+0>' \
          -tile 1x3  ${IMG}_merged.png
          #-pointsize 30        

#### fused EM
IMG=bulkEM
TYPE=bulk
convert ${IMG}_${TYPE}_0.png -geometry 640x A.png
convert ${IMG}_${TYPE}_30.png -geometry 640x B.png
convert ${IMG}.png -geometry 640x C.png

  montage         A.png \
                  B.png \
                  C.png \
          -geometry '+0+0+0+0>' \
          -tile 1x3  ${IMG}_merged.png
          #-pointsize 30        


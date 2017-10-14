IMG=fusedEM
convert ${IMG}_fused_0.png -geometry 640x A.png
convert ${IMG}_fused_30.png -geometry 640x B.png
convert ${IMG}.png -geometry 640x C.png

  montage         A.png \
                  B.png \
                  C.png \
          -geometry '+0+0+0+0>' \
          -tile 1x3  ${IMG}_merged.png
          #-pointsize 30        


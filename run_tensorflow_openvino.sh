
#!/bin/sh

DEVICE=$1

python python/classification_ovtf_histology.py -m "./results/xeon/ice_lake/frozen_histology.pb" \
                       -i "x" \
                       -o "Identity" \
                       -ip "./data/A8D0_CRC-Prim-HE-10_002c.tif_Row_1_Col_451.jpg" \
                       -it "image" \
                       -l "./data/labels_histology.txt" \
                       -f "openvino" \
                       --input_height 150 \
                       --input_width 150 \
                       --input_mean 3 \
                       -d $DEVICE 

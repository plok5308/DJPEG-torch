#!/bin/bash

FILEID="13sWjSLVTpLftRO3f0wi_ZhupgtgIDBUC"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${FILEID}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${FILEID}" -o "jpeg_data.tar.gz"

#curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=FILEID" > /dev/null
#curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=FILEID" -o FILDNAME

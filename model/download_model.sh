#!/bin/bash

FILEID="1OYAqDGovzPn8qTL1zEesbheByt5DUi7o"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${FILEID}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${FILEID}" -o "djpegnet.pth"

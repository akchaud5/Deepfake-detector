# Attention_Network_for_Deepfake_Detection
_This implementation borrows heavily from RECCE model._

### Introduction
This repository contains implementation of a model- _Attention Network for Deepfake Detection_. The input for the model is taken in the form of extracted images from vedios or simply images. The model is PyTorch based. The model achieves Accuracy of 98.12%  and AUC-ROC of  0.998 on Dataset - CelebDF(v2). 

###Preprocessing Data
The model takes data in the form of images- preferrably of uniform size. If the data is in the form of vedios, images must be extracted using various preexisting models, this implementation uses RetinaFace to extract images of uniform size and uniformly throughout the temporal interval of the vedio respectively.

###Dataset
The model could be trained on various datasets like CelebDF, FaceForencis++, DFDC, etc. 
-This repository contains configurations for CelebDF and DFDC dataset. 

###Training and Testing
The model is trained on CelebDF-v2 dataset and tested on CelebDF-v2 and DFDC datasets(i.e training and testing on same dataset and training and testing on different datasets.)




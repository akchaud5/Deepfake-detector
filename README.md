# Attention_Network_for_Deepfake_Detection
_This implementation borrows heavily from RECCE model._

### Introduction
This repository contains implementation of a model- _Attention Network for Deepfake Detection_. The input for the model is taken in the form of extracted images from vedios or simply images. The model is based on PyTorch. The model achieves Accuracy of 98.12%  and AUC-ROC of  0.998 on Dataset - CelebDF(v2). 

### Architecture
We propose a model which identifies features important for detection of forgery in  RGB-spatial domain as well as the frequency domain. Hence we propose to use frequency filters so that minute instances of frequency domain can be used for detection of forgery inspired by works like model M2TR. 

For spatial domain(RGB): We use the Xception model as a baseline. Input image -RGB with white noise is fed to the encoder. The aim is to learn a robust representation of real faces. Since forgery faces could be based on varied methods, learning a constrained representation of forged faces would not help our case. Reconstruction loss is calculated along with metric loss. Metric learning loss is used for each decoder and output of the last encoder. 

For frequency Domain: The embedding of encoder after a few layers are applied with 2DFFT along spatial dimensions. We achieve spectrum representation as a result.

CMF block: The embedding from spectrum representation is fused using cross modality fusion block(CMF).  It consists of first feeding the embeddings of encoder-decoder  through 1X1 convolution individually. Then they are flattened along spatial dimensions to obtain 2D embedding and a fused feature is calculated. Further we apply 3X3 convolution along with residual connection.
The resultant features  are used for forgery detection both in spatial  as well as frequency domain. We further stack 4 CMF blocks on top of each other. Then a simple classification layer is used for obtaining discrete output. 


### Preprocessing Data
The model takes data in the form of images- preferrably of uniform size. If the data is in the form of vedios, images must be extracted using various preexisting models, this implementation uses RetinaFace to extract images of uniform size and uniformly throughout the temporal interval of the vedio respectively.

### Dataset
The model could be trained on various datasets like CelebDF, FaceForencis++, DFDC, etc. 
-This repository contains configurations for CelebDF and DFDC dataset. 

### Training and Testing
The model is trained on CelebDF-v2 dataset and tested on CelebDF-v2 and DFDC datasets(i.e training and testing on same dataset and training and testing on different datasets.)




---
name: DDFD AlexNet Model
caffemodel: --//--
caffemodel_url: --//--
license: unrestricted
---

This model is a replication of the model described in the [Multi-view Face Detection Using Deep Convolutional
Neural Networks](http://arxiv.org/pdf/1502.02766v3.pdf) publication.

Differences from [AlexNet](https://github.com/DolotovEvgeniy/face-detection-model/tree/master/bvlc_alexnet) model:
- Converted the fullyconnected
layers into convolutional layers by reshaping layer
parameters

How to train network:
- Run caffe/data/ilsvrc12/get_ilsvrc_aux.sh (Download imagenet_mean.binaryproto)
- Create new folder caffe/models/bvlc_alexnet_ddfd
- Download trained [AlexNet model](https://drive.google.com/drive/folders/0B6q4BSmVJim6Yl9qT0YyQ0FIZW8) (bvlc_alexnet.caffemodel). Copy it to caffe/models/bvlc_alexnet_ddfd
- Create new folder caffe/examples/ddfd_alexnet
- Copy model and solver to caffe/examples/ddfd_alexnet
- Edit paths to files containing paths to images and labels
- Copy .sh file to Caffe folder
- Run .sh :)
This model was trained by --//--





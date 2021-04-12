# Image Segmentation

## Models

* FCN32
* VGG Segnet
* VGG U-Net

[Graphs Models](https://github.com/reymondzzzz/semantic_segmentation/tree/master/graph_models)

## Getting Started

### Prerequisites

* [OpenCV](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html)
or "sudo apt-get install python-opencv"
* [Theano](http://theano.readthedocs.io/en/latest/install.html)
* [Keras](https://keras.io/#installation)


### Preparing the data for training

You need to make two folders

*  Images Folder - For all the training images
* Annotations Folder - For the corresponding ground truth segmentation images

The filenames of the annotation images should be same as the filenames of the RGB images.

The size of the annotation image for the corresponding RGB image should be same.

For each pixel in the RGB image, the class label of that pixel in the annotation image would be the value of the blue pixel.

## Downloading the Pretrained VGG Weights

You need to download the pretrained VGG-16 weights trained on imagenet

```shell
cd data
wget "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5"
```



## Training the Model

To train the model run the following command:

```shell
python  train.py \
 --save_weights_path=model \
 --train_images="data/dataset1/images_prepped_train/" \
 --train_annotations="data/dataset1/annotations_prepped_train/" \
 --val_images="data/dataset1/images_prepped_test/" \
 --val_annotations="data/dataset1/annotations_prepped_test/" \
 --n_classes=10 \
 --epochs=5 \
 --input_height=320 \
 --input_width=640 \
 --model_name="vgg_segnet"
```

Choose model_name from vgg_segnet  vgg_unet, fcn32

## Getting the predictions

To get the predictions of a trained model

```shell
python  predict.py \
 --save_weights_path=model \
 --test_images="data/dataset1/images_prepped_test/" \
 --output_path="data/predictions/" \
 --n_classes=10 \
 --input_height=320 \
 --input_width=640 \
 --model_name="vgg_segnet"
```

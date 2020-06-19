# IC670 Deep Learning Final Programming Assignment


**DISCLAIMER**
This project may contains enormous amount of bugs. If you want to use this repo, please be aware...

## Purpose of this project
Design a image-captioning network WITHOUT A FRAMEWORK

## Features
Neural Network Layers
 - 2D Convolution
 - Linear
 - 2D Max Pooling
 - RNN (which is not supported yet)

Activation Layers
 - ReLU
 - Softmax
 - Tanh (Not working properly)

Etc Layers
 - Flatten (Reshapes network)
 - Add (Add two inputs to one network)
 - Concat (Concat two images to one)

## Requirements
 - numpy (of course)
 - Every other packages that code saids you have to install...

## Files
 - `main.py` : main network execution
 - `main_cnn.py` : partial network (CNN) execution. It trains simple mnist
 - `load.py` : Loading Flickr8k, glove6B
 - `args.py` : Parsing arguments
 - `image.py` : Image augmentation
 - `log.py` : Quite neat logger. Maybe the only useful file in this repo...
 - `model.py` : model of the main network
 - `./nn/` : Neural network Layers implementations
 - `./skeleton/` : Skeleton code given from Prof. Lim. It processes the work embedding vector

## Execution
### Main network
exectuing `python main.py` will work...

### Sub network (CNN)
`python main_cnn.py -m LeeNetv2 -e l -lr 0.01 (or 0.005)


# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers


The process behind converting custom layers involves...

Ans: 

There are diffrent processes for converting custom layers depending on the framework.

For Caffe:
1. Register custom layer as extension in model optimizer OR
2. Register custom layer as Custom and use system's caffe to calculate the output shape of each Custom Layer

For Tensorflow:
1. Register custom layer as extension in model optimizer OR
2. You need some sub graph that shoud not be in IR and also have another subgraph for that operation. Model Optimizer provides such solution as "Sub Graph Replacement" OR
3. Pass the custom operation to Tensorflow to handle during inference.

For MXNet:
1. MXNet's process is same as tensorflow one. It only not supporting the offloading the custom layer to MXNet to handle.

Some of the potential reasons for handling custom layers are...

Ans: 

The Custom layes known as per their name "Custom" means modified or new. 
There are variety of frameworks which are used for training the deep learning models such as <i>Keras, Tensorflow, ONNX, Caffe etc.</i>

All these frameworks have their own methods to process the tensors (Data) so it may possible that
some functions are not available or behaves diffrently in each other.

Model Optimizer query each layer of trained model from the list of known layers (Supported layers) before building the model's internal representation.
It also optimizes the model by following three steps. Quantization, Freezing and Fusing. At last it generated the intermidiate representation from the trained model. 


## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Each of these use cases would be useful because...

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

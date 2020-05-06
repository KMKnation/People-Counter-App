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

Hence Custom Layer support neccessary for Model Optimizer so that the unsupported operations can be supported through dependent framework during runtime inference.

Model Optimizer query each layer of trained model from the list of known layers (Supported layers) before building the model's internal representation.
It also optimizes the model by following three steps. Quantization, Freezing and Fusing. At last it generated the intermidiate representation from the trained model. 


## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...
- Comparing the size of both models
- Comparing the accuracy of models
- Comparing the inference time of both models

Please checkout the [model_comparison.py](./model_comparison.py) that does the above things.

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

Parameters | pre-conversion | post-conversion
| ------------- | ------------- | -------------
accuracy  | 0.65138817  | 0.6267369
size  | 69.7 MB  | 67.3 MB
inference time  | 3528.266 ms  | 36.44 ms


## Assess Model Use Cases

Some of the potential use cases of the people counter app are...
1. Automate Room Counting
2. Live Audience Statistics
3. Pantry or Canteen space counting
4. Crowd Detection (specifically right now for Covid-19)
5. Security Systems

Each of these use cases would be useful because...
1. Automate Room Counting - This will helps the house owners who rent their house/PG, They can get to know about the people gathering in their house on daily basis.
2. Live Audience Statistics - In event management companies, counting each person is very assumption based task so each output is approx and diffrent with other narrators. With this solution, they only need to process CCTV footage and produce the actual statistics.
3. Pantry or Canteen space counting - In our company, there are many canteens but the problem is that it is occupied everytime whenever i go there. So if i integrate this solution in my company then i can able to know when to go in canteen :D
4. Crowd Detection -  Right now in India, there are many people who are not understanding the current situation that how crucial it is and we have to stay at home, but some people ignoring this going out for chill in name of essential buying, So i think to detect the large crowd gathering in this time is one of the best application.
5. Security Systems - To provide the statistics to building owner that how many people comes in and how many gets out.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...
- In bad light, model sometimes not detecting the person so to solve this we need to either keeping one light on with th Camera or need to build one robust model from scratch.
- Model Accuracy is also a vital part for some cases, specifically in concern to security. If end user agrees whith the lower accuracy and happy with the results and speed then it is best option to use openvino.
- Camera focal length/Image Size. Camera focal length is one of the important parameter to get the desired results without it model can not detect the person if he/she is so far from camera.
# DL-Chroma-Keying


Green screens are a very popular tool in the film and television industry, 
but also in amateur applications, as they facilitate the cropping of images or videos 
when parts of an image are to be replaced by something else. 
However, unwanted color effects caused by the green screen can easily occur (color spill and mixed pixels). 
The manual effort required to crop and remove color effects increases with the required quality. 
It is therefore of interest to automate this process. 

In my master thesis, I investigate how a deep leaning model can solve the 
green screen challenges by transfer learning from existing synthetic green screen images to 
real green screen images. 

To this end, I propose a model that specifically addresses the problems of 
alpha matting, color spill, and mixed pixels via sub-target learning. 
To evaluate different models, I create a green screen benchmark that contains real green screen images. 
In my experiments, the proposed model was able to significantly improve 
the previous performance of existing models, and I was able to gain insights into the desired 
characteristics of a synthetic green screen dataset. 

### Benchmark Dataset

The green screen benchmark contains images of objects standing in front of a green screen. 
(Additional background colors are also available: blue, red, white, black.) 
I created the ground truth of the alpha map using [triangulation matting](https://graphics.stanford.edu/courses/cs148-09-fall/papers/smith-blinn.pdf). 
In the repository (_benchmark_), the code for the automated acquisition of the images for the triangulation 
matting is available. Besides, there is also the code needed to compute the ground truth 
and to complete the dataset.



### Deep Learning Model

This project provides the code that was created in the context of this work (_src_) . Included are: 
- analysis and preprocessing of the training images
- implementation of the model(s) in tensorflow
- the training pipeline with detailed configuration options
- the evaluation of the models

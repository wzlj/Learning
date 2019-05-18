# ResNet V1

## reference
https://blog.csdn.net/liangyihuai/article/details/79140481 

https://github.com/taki0112/ResNet-Tensorflow

## Resnet18 V1 for classification

Classification for small datasets using slim of Tensorflow

### Usage

To run the classification demo, you just follow these steps:

Link to the model:

    $ ln -s /home/swshare/model/ResNet18_v1/* ./models/

Run the network to test classification

    $ python predict.py  --models models/best_models.ckpt 

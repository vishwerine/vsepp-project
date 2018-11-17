#! /bin/bash

STRING="----Welcome to automation script for your programs----"

echo $STRING


STRING="-------Now running VSEPP Model with 0.2 dropout on mini csdv2 Dataset ---------------"
echo $STRING

python train.py --max_violation --logger_name=runs/resnet152_drop0.2 --dropout_value=0.2 --num_epochs=300 --cnn_type='resnet152'  



STRING="-------Now running VSEP Model with no dropout on minicsdv2 resnet152---------------"
echo $STRING
python train.py --max_violation --logger_name=runs/resnet152_no_drop --dropout_value=0 --num_epochs=300 --cnn_type='resnet152'  




STRING="-------Now running VSEP Model with no dropout on minicsdv2 vgg19---------------"
echo $STRING

python train.py --max_violation --logger_name=runs/vgg19_no_drop --dropout_value=0 --num_epochs=300 --cnn_type='vgg19'  



STRING="-------Now running VSEP Model with dropout 0.2 on minicsdv2 vgg19---------------"
echo $STRING

python train.py --max_violation --logger_name=runs/vgg19_drop0.2 --dropout_value=0.2 --num_epochs=300 --cnn_type='vgg19'  




STRING="-------END OF SCRIPT -thank you so much for waiting---------------"
echo $STRING

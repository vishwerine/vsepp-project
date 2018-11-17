#! /bin/bash

STRING="----Welcome to automation script for your programs----"

echo $STRING


STRING="-------Now running VSEPP Model with 0.2 dropout on Flickr30k Dataset ---------------"
echo $STRING
mkdir runs/runX/f30k_precomp_drop0.2
python train.py --max_violation --logger_name=runs/runX/f30k_precomp_drop0.2 --dropout_value=0.2 --num_epochs=300 --data_name='f30k_precomp' --data_path='/home/dcsaero01/data/datasets/vsepp/'



STRING="-------Now running VSEP Model with no dropout on coco resnet152---------------"
echo $STRING
mkdir runs/runX/coco_resnet152_no_drop
python train.py --max_violation --logger_name=runs/runX/coco_resnet152_no_drop --dropout_value=0 --num_epochs=50 --data_name='coco' --data_path='/home/dcsaero01/data/projects/vsepp/data/'



STRING="-------Now running VSEP Model with no dropout on coco resnet152---------------"
echo $STRING
mkdir runs/runX/coco_resnet152_drop0.2
python train.py --max_violation --logger_name=runs/runX/coco_resnet152_drop0.2 --dropout_value=0.2 --num_epochs=50 --data_name='coco' --data_path='/home/dcsaero01/data/projects/vsepp/data/'



STRING="-------END OF SCRIPT -thank you so much for waiting---------------"
echo $STRING

import pickle
import os
import time
import shutil

import torch

import data
from vocab import Vocabulary  # NOQA
from precomp_main_author import VSEModel
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data

import logging

import numpy as np

import argparse
import sklearn.preprocessing



def fetch_caption(index,cap_emb,img_emb):

    scores = []
    for i in range(img_emb.shape[0]):
        scores.append(np.dot(img_emb[index],cap_emb[i]))

    scores  = np.asarray(scores)

    return np.argsort(scores)[-5:]


def fetch_image(index,cap_emb,img_emb):

    scores = []
    for i in range(cap_emb.shape[0]):
        scores.append(np.dot(cap_emb[index],img_emb[i]))

    scores  = np.asarray(scores)

    return np.argsort(scores)[-5:][-1:]

import random

def random_fetch_image(index,cap_emb,img_emb):

    return random.randint(0,img_emb.shape[0]-1)

def main(opt):

    # Construct the model
    model = VSEModel(opt)

    #print('main')

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            
            
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            
            print("=> loaded checkpoint")
            #validate(opt, val_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))


    dirname = '/home/vishwash/data/datasets/minicsd/embedproject/'
    images = np.load(dirname+'preds_train.npy')
    captions = np.load(dirname+'newvecs_train.npy')

    images  = sklearn.preprocessing.normalize(images)
    captions  =sklearn.preprocessing.normalize(captions)


    images = torch.Tensor(images)
    captions = torch.Tensor(captions)

    img_emb, cap_emb = model.forward_emb(images, captions)

    images2 = np.load(dirname+'preds_test.npy')
    captions2 = np.load(dirname+'newvecs_test.npy')

    images2 = torch.Tensor(images2)
    captions2 = torch.Tensor(captions2)

    img_emb2, cap_emb2 = model.forward_emb(images2, captions2)




    img_emb = img_emb.cpu().data.numpy()
    cap_emb = cap_emb.cpu().data.numpy()


    img_emb2 = img_emb2.cpu().data.numpy()
    cap_emb2 = cap_emb2.cpu().data.numpy()



    count = 0
    for i in range(cap_emb.shape[0]):
        if i in fetch_image(i,cap_emb, img_emb):
            count = count+1

    print(count*1.0/cap_emb.shape[0])





class Opt:
    batch_size = 25
    workers = 10
    img_dim = 4096
    text_dim = 5100
    embed_size = 1024
    margin = 0.2
    measure = 'cosine'
    learning_rate = 0.0002
    max_violation = True
    num_epochs = 30
    resume = '/home/vishwash/data/projects/vsepp/authormodel/299model_best'

opt =  Opt()


main(opt)



import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import sklearn.preprocessing

from collections import OrderedDict
import shutil

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
    num_epochs = 300

opt =  Opt()


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class PrecompCSDDataset(data.Dataset):
 
    def __init__(self):

        dirname = '/home/vishwash/data/datasets/minicsd/embedproject/'
        self.images = np.load(dirname+'preds_train.npy')
        self.captions = np.load(dirname+'newvecs_train.npy')
        self.images = sklearn.preprocessing.normalize(self.images)
        self.captions = sklearn.preprocessing.normalize(self.captions)
        #scaler = MinMaxScaler()
        #self.images = scaler.fit_transform(self.images.tolist())
        #self.captions = scaler.fit_transform(self.captions.tolist())

    def __getitem__(self, index):
        img_id = index
        image = torch.Tensor(self.images[img_id])
        target = torch.Tensor(self.captions[img_id])

        return image, target, index, img_id

    def __len__(self):
        return int(self.images.shape[0])



def get_precomp_loader( opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompCSDDataset()

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True)
    return data_loader




class EncoderImage(nn.Module):

    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage, self).load_state_dict(new_state)



class EncoderText(nn.Module):

    def __init__(self, text_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        self.fc = nn.Linear(text_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, captions):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(captions)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderText, self).load_state_dict(new_state)


def cosine_sim(im, s):
        """Cosine similarity between all the image and sentence pairs
    
        """

        return im.mm(s.t())



class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim

        self.max_violation = max_violation

    

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()



class VSEModel(object):

    def __init__(self, opt):

        self.img_enc = EncoderImage(opt.img_dim ,opt.embed_size)
        self.txt_enc = EncoderText( opt.text_dim, opt.embed_size)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         measure=opt.measure,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())

        self.params = params

        #print(len(params))

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

  

    def forward_emb(self, images, captions, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions)
        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb)

        #self.logger.update('Le', loss.data[0], img_emb.size(0))
        return loss

    def train_emb(self, images, captions, ids=None):
        """One training step given images and captions.
        """
        self.Eiters += 1
        #self.logger.update('Eit', self.Eiters)
        #self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, captions)

        #print(self.img_enc.fc.weight.cpu().data.numpy())

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb)

        

        # compute gradient and do SGD step
        loss.backward()
        self.optimizer.step()


train_loader = get_precomp_loader( opt)

model = VSEModel(opt)

#print(model.params)



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')



def train():

    for epoch in range(opt.num_epochs):
        for i, train_data in enumerate(train_loader):

            #print(i)

            images, targets, indices, imgindices = train_data

    
            model.train_emb(images, targets)
            img_emb , cap_emb  =model.forward_emb(images, targets)

            #print(cosine_sim(img_emb,cap_emb))


    save_checkpoint({
            'model':model.state_dict(),
            }, True, 'model_best', 'authormodel/'+str(epoch))


train()


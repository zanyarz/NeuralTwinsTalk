from __future__ import absolute_import, division, print_function

import copy
import json
import os
import random
import numpy as np
import misc.utils as utils
import torch
import torch.utils.data as data
import torchtext.vocab as vocab  # use this to load glove vector
#import bert_embedding as Bert_Embeddings # We use this to load bert vector reporesentations for 
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO   
from six.moves import xrange
import gluonnlp as Gnlp
import mxnet as mx
from bert_embedding import BertEmbedding

from .dataloader_hdf import HDFSingleDataset


class DataLoader(data.Dataset):
    
    def __init__(self, opt, split='train', seq_per_img=5):
        
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img
        self.seq_length = opt.seq_length
        self.split = split
        self.seq_per_img = seq_per_img
        
        # image processing function.
        if split == 'train':
            self.Resize = transforms.Resize((self.opt.image_size, self.opt.image_size))
        else:
            self.Resize = transforms.Resize((self.opt.image_crop_size, self.opt.image_crop_size))

        self.RandomCropWithBbox = utils.RandomCropWithBbox(opt.image_crop_size)
        self.ToTensor = transforms.ToTensor()
        self.res_Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.vgg_pixel_mean = np.array([[[102.9801, 115.9465, 122.7717]]])

        self.max_gt_box = 100
        self.max_proposal = 200
        
        if opt.glove_6B_300:
            self.glove = vocab.GloVe(name='6B', dim=300)
            
        if opt.bert_base_768:
            self.bertmodel, self.bertvocab = Gnlp.model.get_model(name='bert_12_768_12',dataset_name='book_corpus_wiki_en_uncased',pretrained=True,ctx=mx.cpu(),use_pooler=False,use_decoder=False,use_classifier=False)
            self.bertembed = BertEmbedding(ctx=mx.cpu(), dtype='float32', model='bert_12_768_12', dataset_name='book_corpus_wiki_en_uncased', params_path=None, max_seq_length=25, batch_size=25)
            self.bertmodel = None

        if opt.det_oracle == True:
            print('Training and Inference under oracle Mode...')

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_dic)
        self.info = json.load(open(self.opt.input_dic))
        self.itow = self.info['ix_to_word']
        self.wtoi = {w:i for i,w in self.itow.items()}
        self.wtod = {w:i+1 for w,i in self.info['wtod'].items()} # word to detection
        self.dtoi = {w:i+1 for i,w in enumerate(self.wtod.keys())} # detection to index
        self.itod = {i+1:w for i,w in enumerate(self.wtod.keys())}
        self.wtol = self.info['wtol']
        self.ltow = {l:w for w,l in self.wtol.items()}
        self.vocab_size = len(self.itow) + 1 # since it start from 1
        print('vocab size is ', self.vocab_size)

        # initilize the fg+s/p map back to word idx.
        self.st2towidx = np.zeros(len(self.dtoi)*2+1) # statge 2 to word idex
        for w, i in self.dtoi.items():
            s2_idx = i * 2 - 1
            if w not in self.wtoi:
                w = 'UNK'
            w_idx = self.wtoi[w]
            self.st2towidx[s2_idx] = w_idx
            # get the plural idx.
            if w in self.ltow:
                pw = self.ltow[w]
                w_idx = self.wtoi[pw]
            self.st2towidx[s2_idx+1] = w_idx
        

        # get the glove vector for the fg detections.
        if opt.glove_6B_300:
            self.glove_fg = np.zeros((len(self.dtoi)+1, 300))
            for i, word in enumerate(self.dtoi.keys()):
                vector = np.zeros((300))
                count = 0
                for w in word.split(' '):
                    count += 1
                    if w in self.glove.stoi:
                        glove_vector = self.glove.vectors[self.glove.stoi[w]]
                        vector += glove_vector.numpy()
                    else: # use a random vector instead
                        random_vector = 2*np.random.rand(300) - 1
                        vector += random_vector
                self.glove_fg[i+1] = vector / count

            self.glove_w = np.zeros((len(self.wtoi)+1, 300))
            for i, word in enumerate(self.wtoi.keys()):
                vector = np.zeros((300))
                count = 0
                for w in word.split(' '):
                    count += 1
                    if w in self.glove.stoi:
                        glove_vector = self.glove.vectors[self.glove.stoi[w]]
                        vector += glove_vector.numpy()
                    else: # use a random vector instead
                        random_vector = 2*np.random.rand(300) - 1
                        vector += random_vector
                self.glove_w[i+1] = vector / count

        #if opt.bert_base_ctx:
        #    self.bert_ctx_fg = np.zeros((len(self.dtoi)+1, 768))
        #    # get the bert vector for the fg detections.
        
        #    for i, word in enumerate(self.dtoi.keys()):
        #        vector = np.zeros((768))
        #        count = 0
        #        sent_bert = self.bertembed(word)
        #        for w in word.split(' '):
        #            count += 1
        #            if w in self.bertvocab.token_to_idx:
        #                temp = self.bertembed(w)
        #                bert_vec = temp[0][1][0]
        #                vector += np.array(bert_vec)
        #            else:
        #                random_vector = 2*np.random.rand(768) - 1
        #                vector += random_vector
#
        #        self.bert_ctx_fg[i+1] = vector / count

        #    self.bert_w = np.zeros((len(self.wtoi)+1, 768))
        #    for i, word in enumerate(self.wtoi.keys()):
        #        vector = np.zeros((768))
        #        count = 0
        #        for w in word.split(' '):
        #            count += 1
        #            if w in self.bertvocab.token_to_idx:
        #                temp = self.bertembed(w)
        #                bert_vec = temp[0][1][0]
        #                vector += np.array(bert_vec)
        #            else:
        #                random_vector = 2*np.random.rand(768) - 1
        #                vector += random_vector
        #        self.bert_ctx_w[i+1] = vector / count

        # get the bert vector for the fg detections.
        if opt.bert_base_768:
            self.bert_fg = np.zeros((len(self.dtoi)+1, 768))
            for i, word in enumerate(self.dtoi.keys()):
                vector = np.zeros((768))
                count = 0
                for w in word.split(' '):
                    #count += 1
                    if w in self.bertvocab.token_to_idx:
                        temp = self.bertembed(w)
                        bert_vec = temp[0][1][0]
                        vector += np.array(bert_vec)
                    #else:
                        random_vector = 2*np.random.rand(768) - 1
                        vector += random_vector
                self.bert_fg[i+1] = vector

            self.bert_w = np.zeros((len(self.wtoi)+1, 768))
            for i, word in enumerate(self.wtoi.keys()):
                vector = np.zeros((768))
                count = 0
                for w in word.split(' '):
                    #count += 1
                    if w in self.bertvocab.token_to_idx:
                        temp = self.bertembed(w)
                        bert_vec = temp[0][1][0]
                        vector += np.array(bert_vec)
                    else:
                        random_vector = 2*np.random.rand(768) - 1
                        vector += random_vector
                self.bert_w[i+1] = vector

        # open the caption json file
        print('DataLoader loading json file: ', opt.input_json)
        self.caption_file = json.load(open(self.opt.input_json))

        # open the detection json file.
        self.dataloader_hdf = HDFSingleDataset(self.opt.proposal_h5)

        # load the coco grounding truth bounding box.
        det_train_path = '%s/coco/annotations/instances_train2014.json' %(opt.data_path)
        det_val_path = '%s/coco/annotations/instances_val2014.json' %(opt.data_path)

        self.coco_train = COCO(det_train_path)
        self.coco_val = COCO(det_val_path)

        # category id to labels. +1 becuase 0 is the background label.
        self.ctol = {c:i+1 for i, c in enumerate(self.coco_val.cats.keys())}
        self.itoc = {i+1:c['name'] for i, c in enumerate(self.coco_val.cats.values())}
        self.ctoi = {c:i for i, c in self.itoc.items()}



        if opt.bert_base_768:
            self.bert_clss = np.zeros((len(self.itoc)+1, 768))
            for i, word in enumerate(self.itoc.values()):
                vector = np.zeros((768))
                count = 0
                # if we decode novel word, replace the word representation based on the dictionary.
                if opt.decode_noc and word in utils.noc_word_map:
                    word = utils.noc_word_map[word]

                for w in word.split(' '):
                    #count += 1
                    if w in self.bertvocab.token_to_idx:
                        temp = self.bertembed(w)
                        bert_vec = temp[0][1][0]
                        vector += np.array(bert_vec)
                    else:
                        random_vector = 2*np.random.rand(768) - 1
                        vector += random_vector
                self.bert_clss[i+1] = vector



        if opt.glove_6B_300:
            self.glove_clss = np.zeros((len(self.itoc)+1, 300))
            for i, word in enumerate(self.itoc.values()):
                vector = np.zeros((300))
                count = 0
                # if we decode novel word, replace the word representation based on the dictionary.
                if opt.decode_noc and word in utils.noc_word_map:
                    word = utils.noc_word_map[word]

                for w in word.split(' '):
                    count += 1
                    if w in self.glove.stoi:
                        glove_vector = self.glove.vectors[self.glove.stoi[w]]
                        vector += glove_vector.numpy()
                    else: # use a random vector instead
                        random_vector = 2*np.random.rand(300) - 1
                        vector += random_vector
                self.glove_clss[i+1] = vector / count



        self.detect_size = len(self.ctol)
        self.fg_size = len(self.dtoi)
        # get the fine-grained mask.
        self.fg_mask = np.ones((self.detect_size+1, self.fg_size+1))
        for w, det in self.wtod.items():
            self.fg_mask[det, self.dtoi[w]] = 0

        # separate out indexes for each of the provided splits
        self.split_ix = []
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == split:
                self.split_ix.append(ix)
        print('assigned %d images to split %s' %(len(self.split_ix), split))

    def get_det_word(self, gt_bboxs, captions, ngram=2):
        # get the present category.
        pcats = []
        for i in range(gt_bboxs.shape[0]):
            pcats.append(gt_bboxs[i,4])

        # get the orginial form of the caption.
        indicator = []
        stem_caption = []
        for s in captions:
            tmp = []
            for w in s:
                tmp.append(self.wtol[w])
            stem_caption.append(tmp)
            indicator.append([(0, 0, 0)]*len(s)) # category class, binary class, fine-grain class.

        ngram_indicator = {i+1:copy.deepcopy(indicator) for i in range(ngram)}
        # get the 2 gram of the caption.
        for n in range(ngram,0,-1):
            for i, s in enumerate(stem_caption):
                for j in xrange(len(s)-n+1):
                    ng = ' '.join(s[j:j+n])
                    # if the n-gram exist in word_to_detection dictionary.
                    if ng in self.wtod and indicator[i][j][0] == 0 and self.wtod[ng] in pcats: # make sure that larger gram not overwright with lower gram.
                        bn = (ng != ' '.join(captions[i][j:j+n])) + 1
                        fg = self.dtoi[ng]
                        ngram_indicator[n][i][j] = (self.wtod[ng], bn, fg)
                        indicator[i][j:j+n] = [(self.wtod[ng], bn, fg)] * n
        return ngram_indicator

    def __getitem__(self, index):
        
        ix = self.split_ix[index]

        # load image here.
        image_id = self.info['images'][ix]['id']
        file_path = self.info['images'][ix]['file_path']

        proposal_item =copy.deepcopy(self.dataloader_hdf[ix])
        num_proposal = int(proposal_item['dets_num'])
        num_nms = int(proposal_item['nms_num'])
        proposals = proposal_item['dets_labels']
        proposals = proposals.squeeze()[:num_nms, :]

        coco_split = file_path.split('/')[0]
        # get the ground truth bounding box.
        if coco_split == 'train2014':
            coco = self.coco_train
        else:
            coco = self.coco_val

        bbox_ann_ids = coco.getAnnIds(imgIds=image_id)
        bbox_ann = [{'label': self.ctol[i['category_id']], 'bbox': i['bbox']} for i in coco.loadAnns(bbox_ann_ids)]

        gt_bboxs = np.zeros((len(bbox_ann), 5))
        for i, bbox in enumerate(bbox_ann):
            gt_bboxs[i, :4] = bbox['bbox']
            gt_bboxs[i, 4] = bbox['label']

        # convert from x,y,w,h to x_min, y_min, x_max, y_max
        gt_bboxs[:,2] = gt_bboxs[:,2] + gt_bboxs[:,0]
        gt_bboxs[:,3] = gt_bboxs[:,3] + gt_bboxs[:,1]

        # load the image.
        img = Image.open(os.path.join(self.opt.image_path, file_path)).convert('RGB')

        width, height = img.size
        # resize the image.
        img = self.Resize(img)

        if self.split == 'train':
            # resize the gt_bboxs and proposals.
            proposals = utils.resize_bbox(proposals, width, height, self.opt.image_size, self.opt.image_size)
            gt_bboxs = utils.resize_bbox(gt_bboxs, width, height, self.opt.image_size, self.opt.image_size)
        else:
            proposals = utils.resize_bbox(proposals, width, height, self.opt.image_crop_size, self.opt.image_crop_size)
            gt_bboxs = utils.resize_bbox(gt_bboxs, width, height, self.opt.image_crop_size, self.opt.image_crop_size)

        # crop the image and the bounding box.
        img, proposals, gt_bboxs = self.RandomCropWithBbox(img, proposals, gt_bboxs)

        gt_x = (gt_bboxs[:,2]-gt_bboxs[:,0]+1)
        gt_y = (gt_bboxs[:,3]-gt_bboxs[:,1]+1)
        gt_area_nonzero = (((gt_x != 1) & (gt_y != 1)))

        gt_bboxs = gt_bboxs[gt_area_nonzero]
        captions = self.caption_file[ix]

        # given the bbox_ann, and caption, this function determine which word belongs to the detection.
        det_indicator = self.get_det_word(gt_bboxs, captions)

        # fetch the captions
        ncap = len(captions) # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        # convert caption into sequence label.
        cap_seq = np.zeros([ncap, self.seq_length, 5])
        for i, caption in enumerate(captions):
            j = 0
            k = 0
            while j < len(caption) and j < self.seq_length:
                is_det = False
                for n in range(2, 0, -1):
                    if det_indicator[n][i][j][0] != 0:
                        cap_seq[i,k,0] = det_indicator[n][i][j][0] + self.vocab_size
                        cap_seq[i,k,1] = det_indicator[n][i][j][1]
                        cap_seq[i,k,2] = det_indicator[n][i][j][2]
                        cap_seq[i,k,3] = self.wtoi[caption[j]]
                        cap_seq[i,k,4] = self.wtoi[caption[j]]

                        is_det = True
                        j += n # skip the ngram.
                        break
                if is_det == False:
                    cap_seq[i,k,0] = self.wtoi[caption[j]]
                    cap_seq[i,k,4] = cap_seq[i,k,0]
                    j += 1
                k += 1

        # get the mask of the ground truth bounding box. The data shape is
        # num_caption x num_box x num_seq
        box_mask = np.ones((len(captions), gt_bboxs.shape[0], self.seq_length))
        for i in range(len(captions)):
            for j in range(self.seq_length):
                if cap_seq[i,j,0] > self.vocab_size:
                    box_mask[i,:,j] = ((gt_bboxs[:,4] == (cap_seq[i,j,0]-self.vocab_size)) == 0)

        # get the batch version of the seq and box_mask.
        if ncap < self.seq_per_img:
            seq_batch = np.zeros([self.seq_per_img, self.seq_length, 4])
            mask_batch = np.zeros([self.seq_per_img, gt_bboxs.shape[0], self.seq_length])
            # we need to subsample (with replacement)
            for q in range(self.seq_per_img):
                ixl = random.randint(0,ncap)
                seq_batch[q,:] = cap_seq[ixl,:,:4]
                mask_batch[q,:]=box_mask[ixl]
        else:
            ixl = random.randint(0, ncap - self.seq_per_img)
            seq_batch = cap_seq[ixl:ixl+self.seq_per_img,:,:4]
            mask_batch = box_mask[ixl:ixl+self.seq_per_img]

        input_seq = np.zeros([self.seq_per_img, self.seq_length+1, 4])
        input_seq[:,1:] = seq_batch

        gt_seq = np.zeros([10, self.seq_length])
        gt_seq[:ncap,:] = cap_seq[:,:,4]
        pad_proposals = np.zeros((self.max_proposal, 6))
        pad_gt_bboxs = np.zeros((self.max_gt_box, 5))
        pad_box_mask = np.ones((self.seq_per_img, self.max_gt_box, self.seq_length+1))

        if self.opt.det_oracle == False:
            num_pps = min(proposals.shape[0], self.max_proposal)
            num_box = min(gt_bboxs.shape[0], self.max_gt_box)
            pad_proposals[:num_pps] = proposals[:num_pps]
            pad_gt_bboxs[:num_box] = gt_bboxs[:num_box]
            pad_box_mask[:,:num_box,1:] = mask_batch[:,:num_box,:]
        else:
            num_pps = min(gt_bboxs.shape[0], self.max_proposal)
            pad_proposals[:num_pps] = np.concatenate((gt_bboxs[:num_pps], np.ones([num_pps,1])),axis=1)
            num_box = min(gt_bboxs.shape[0], self.max_gt_box)
            pad_gt_bboxs[:num_box] = gt_bboxs[:num_box]
            pad_box_mask[:,:num_box,1:] = mask_batch[:,:num_box,:]

        input_seq = torch.from_numpy(input_seq).long()
        gt_seq = torch.from_numpy(gt_seq).long()
        pad_proposals = torch.from_numpy(pad_proposals).float()
        pad_box_mask = torch.from_numpy(pad_box_mask).byte()
        pad_gt_bboxs = torch.from_numpy(pad_gt_bboxs).float()
        num = torch.FloatTensor([ncap, num_pps, num_box])

        if self.opt.cnn_backend == 'vgg16':
            img = np.array(img, dtype='float32')
            img = img[:,:,::-1].copy() # RGB --> BGR
            img -= self.vgg_pixel_mean
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1).contiguous()
        else:
            img = np.array(img, dtype='float32')
            img = img[:,:,::-1].copy() # RGB --> BGR
            img /= 255 # Convert range to [0,1]
            img = self.ToTensor(img)
            img = self.res_Normalize(img)

        return img, input_seq, gt_seq, num, pad_proposals, pad_gt_bboxs, pad_box_mask, image_id

    def __len__(self):
        return len(self.split_ix)

from Algos.TRCA import *
from Algos.TDCA import *
from Algos.ECCA import *
from Algos.Filterbank import *
# from plots import *
from Algos.GZSL import *
from Algos.iGZSL import *

import numpy as np
import time
import random

class CrossValidation:
    """
    Leave-one-block-out (lobo) cross validation on TRCA/TDCA. (N-1) train + 1 test
    Note: in SSVEP-BCI, one block = Nclass trials, i.e., subjects gaze at one target at each trial
    Inputs:
        args: includes blocks, labels, ...
    
    Returns:
        obj

    """

    def __init__(self, args):
        self.args = args


    def separate_all_into_train_test(self, eeg, block_test, blocks):
        """
        Separate whole dataset into training blocks and test blocks
        Inputs
            eeg - ndarray (block slide class chnl tp): whole SSVEP dataset
            block_test - list: 1 test block for lobo
            blocks - list: all blocks

        Returns
            trainblocks / testblocks - dictionary: 
                d['data'] - ndarray (block, slide, class, chnl, tp) # slide=0 for testblocks since test blocks mimic online decoding, which only use [0.14, 0.14+t]s data
                d['label'] - list: all classes [0-39] for both
                d['block'] - list: blocks for training & test
        """
        block_train = [item for item in blocks if item not in block_test]
        block_train_idx = [blocks.index(item) for item in block_train]
        block_test_idx = [blocks.index(item) for item in block_test]
        trainblocks = {
            'data': eeg[block_train_idx],
            'label': self.args.labels,
            'block': block_train,
        }

        testblocks = {
            'data': eeg[block_test_idx,0,:,:,:],
            'label': self.args.labels,
            'block': block_test,
        }

        return trainblocks, testblocks
    

    def separate_into_seen_unseen(self, eeg_train, sc_template):
        """
        Separate training blocks by seen/unseen classes, into seen (for training) and unseen (for testing/analysis potentially)
        Inputs
            eeg_train - ndarray (block slide class chnl tp): eeg data of training blocks
            sc_template - ndarray (slide class 2Nh tp): sine-cosine template of all frequencies

        Returns:
            seen/unseen - dictionary:
                d['data'] - ndarray (block, slide, Nc_seen/Nc_unseen, chnl, tp): eeg data, all training blocks, but part classes
                d['sc_template'] - ndarray(block, Nc_seen/Nc_unseen, chnl, tp): sine-cosine template, part classes
                d['label'] - list: part of classes in [0-39] based on self.label_unseen
        """
        
        seen = {
            'data': eeg_train[:, :, self.args.label_seen, :, :],
            'sc_template': sc_template[:, self.args.label_seen, :, :],
            'label': self.args.label_seen,
        }
        unseen = {
            'data': eeg_train[:, :, self.args.label_unseen, :, :],
            'sc_template': sc_template[:, self.args.label_unseen, :, :],
            'label': self.args.label_unseen,
        }
        return seen, unseen

    
    def separate_into_train_valid(self, seen):
        """
        Separate seen data by classes (self.label_validate), into train data and validate data (class separation, not block)
        Inputs
            seen - dictionary: seen classes data, sc_template, label in training blocks
        Returns
            trainset/validset - dictionary
            d['label'] - list: part of classes in seen classes, based on self.label_valid
            d['data'] - ndarray (blocks,slide,Nc_train/Nc_validate,chnl,tp): data of part of classes in seen classes data
            d['sc_template'] - ndarray (slide,Nc_train/Nc_validate,chnl,tp): sine-cosine template of part of classes in seen classes
        
        Note: since class dimension (dim=2) in seen['data'] is not 40, but Nc_seen, and train/valid label is in seen classes, the index of train/valid classes should be found first, rather than directly seen['data'][:,:,self.label_valid,:,:].
        """
        train_in_seen_idx = [self.args.label_seen.index(item) for item in self.args.label_train]
        valid_in_seen_idx = [self.args.label_seen.index(item) for item in self.args.label_valid]

        trainset = {
            'data': seen['data'][:,:,train_in_seen_idx,:,:],
            'sc_template': seen['sc_template'][:,train_in_seen_idx,:,:],
            'label': self.args.label_train,
        }
        validset = {
            'data': seen['data'][:,:,valid_in_seen_idx,:,:],
            'sc_template': seen['sc_template'][:,valid_in_seen_idx,:,:],
            'label': self.args.label_valid,
        }
        return trainset, validset


    def get_norm_params(self, trainset, validset):
        eeg = np.concatenate((trainset['data'], validset['data']), axis=2) # block slide, Nc, chnl, tp
        mean, std = np.mean(eeg), np.std(eeg)
        return mean, std

    
    def normalize_eeg(self, mean, std, dataset):
        dataset['data'] = (dataset['data'] - mean) / std
        return dataset


    def calculate_spatialfilters(self, data, model='trca'):
        """
        Calculate spatial filter using data (should be from seen classes in training blocks)
        Inputs:
            data - ndarray (block,class,chnl,tp): data for SF calculation (slide=0)

        Returns
            weights_sf - matrix (class,chnl): multiple Nch*1 SF using TRCA
        """
        if model == 'ecca':
            return None
        
        if len(data.shape) != 4:
            raise ValueError("Spatial filter calculation: Input data might have slides dim. ")

        ## Calculate spatial filters
        weights_sf = [] 
        if model == 'trca':
            ## TRCA
            model_sf = TRCA()
            data_input = data
            for class_i in range(data_input.shape[1]):
                eeg = data_input[:, class_i, :, :]
                weights_sf.append(model_sf._trca(eeg))
            weights_sf = np.stack(weights_sf, axis=0) # class_seen_num * channels
        elif model == 'tdca':
            model_sf = TDCA(self.args.frequencies, self.args.phases, int(self.args.window*self.args.sampling_rate), self.args.sampling_rate, fb_num=1, 
                            delay_num=1)
            weights_sf = model_sf._tdca(np.transpose(data, [-1, -2, 0, 1])) # 
            weights_sf = np.transpose(weights_sf, [-1, -2]) # subspace chnl
        elif model == 'none':
            weights_sf = np.identity(data.shape[-2]) # chnl chnl
        else:
            raise ValueError("preproc-sf not found!")

        return weights_sf

    
    # def cosine_embedding_loss(self, X, Y, eps=1e-6):
    #     # input: (batch, 1, 1, Ns)
    #     assert X.shape == Y.shape
    #     x_mean, y_mean = torch.mean(X, dim=-1), torch.mean(Y, dim=-1)
    #     X = X - x_mean
    #     Y = Y - y_mean
    #     corrs = torch.sum(X*Y, dim=-1) / (torch.norm(X, dim=-1) * torch.norm(Y, dim=-1)+eps) # (batch, 1, 1)
    #     # corr_tmp = np.corrcoef(X[:X.shape[-1]].detach().cpu().numpy(), Y[:X.shape[-1]].detach().cpu().numpy())
    #     outs = 1 - corrs
    #     out_mean = torch.mean(outs)
    #     return out_mean


    def lobo_tradition(self, eeg_1subj, block_test):
        ## Lobo on traditional TRCA/TDCA methods

        ## Separate dataset
        trainblocks, testblocks = self.separate_all_into_train_test(eeg_1subj, block_test, self.args.blocks)

        ## Extract test eeg and train eeg (trad_trainblocks in (N-1) blocks)
        eeg_train = trainblocks['data'][0:self.args.trad_trainblocks,0,:,:,:] ## blks cls chnl tp
        eeg_test = testblocks['data']
        eeg_train = np.expand_dims(eeg_train, axis=-3)
        eeg_test = np.expand_dims(eeg_test, axis=-3)

        ## Filterbank
        filterbank = Filterbank(self.args.fb_num, self.args.sampling_rate)
        eeg_train_fb = filterbank.fb_filter(eeg_train) # blk cls fb ch tp
        eeg_test_fb = filterbank.fb_filter(eeg_test) # 1 cls fb ch tp

        ## Train model
        if self.args.model == 'trca':
            model = TRCA(self.args.fb_num)
            
        elif self.args.model == 'tdca':
            model = TDCA(self.args.frequencies, self.args.phases, int(self.args.window*self.args.sampling_rate), self.args.sampling_rate, self.args.fb_num)
        else:
            raise ValueError("Model not found!")
        model.train(eeg_train_fb)

        ## Predict
        predicts = []
        for sample_i in range(self.args.class_num):
            test_epoch = eeg_test_fb[0,sample_i,:,:,:] # fb chnl tp
            predict, _ = model.detect(test_epoch)
            predicts.append(predict)
        acc = np.mean(predicts==np.array(self.args.labels))
        print(f'Subject={self.args.subject}, block_test={block_test}, Acc={acc:.4f}')
        metrics = {'loss_train': 0, 'loss_valid': 0, 'loss_unseen': 0,
                   'corr_train': 0, 'corr_valid': 0, 'corr_unseen': 0,
                   'acc': acc, 'predicts': predicts, }
        return metrics
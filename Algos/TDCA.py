import numpy as np
import scipy
# from Utils.References import *
from PrepareData.References import *

"""
Matlab: qr(Mat, 0)?
"""

class TDCA:
    def __init__(self, frequencies, phases, timepoint_num, sampling_rate=250, fb_num=5, subspace_num=8, delay_num=5, harmonic_num=5, fb_coeff_a=-1.25, fb_coeff_b=0.25):
        self.fb_num = fb_num
        self.fb_coeff_a = fb_coeff_a
        self.fb_coeff_b = fb_coeff_b
        self.subspace_num = subspace_num
        self.delay_num = delay_num
        self.frequencies = frequencies
        self.phases = phases
        self.harmonic_num = harmonic_num
        self.sampling_rate = sampling_rate
        self.timepoint_num = timepoint_num

        self.fb_coeffs = np.arange(1, self.fb_num+1)
        self.fb_coeffs = np.power(self.fb_coeffs, self.fb_coeff_a) + self.fb_coeff_b

        self.references = SSVEPReference(self.timepoint_num, self.harmonic_num, self.frequencies, self.sampling_rate, self.phases).get_refer_allfreqs() # class * 2Nh * tp


    def train(self, eeg_fb):
        """
        Input:
            eeg_fb: (blocks, classes, fbs, channels, timepoints)
        """
        self.block_num, self.class_num, fb_num_input, self.channel_num, self.timepoint_num = eeg_fb.shape
        if self.fb_num > fb_num_input:
            raise ValueError("Error: (fb_num of input eeg) < assigned fb_num in classdef.")
        
        eeg_fb = np.transpose(eeg_fb, (4, 3, 0, 2, 1)) # tp ch blk fb class
        

        self.templates = self.update_template(eeg_fb)
        self.weights = self.update_weight(eeg_fb)
        

        a = 1

    def detect(self, eeg_fb):
        # eeg_fb - ndarray - (fbs, channels, tp)

        rho_mat = np.zeros((self.class_num, self.fb_num))
        for fb_ii in range(self.fb_num):
            eeg = eeg_fb[fb_ii,:,:] # ch tp
            for class_ii in range(self.class_num):
                eeg_augment = np.array([])
                for delay_ii in range(self.delay_num):
                    eeg_zeropad = np.concatenate((eeg[:, delay_ii:], np.zeros((self.channel_num, delay_ii))), axis=1)
                    eeg_augment = np.concatenate((eeg_augment, eeg_zeropad),axis=0) if eeg_augment.size else eeg_zeropad

                reference = self.references[class_ii, :,:] # 2Nh tp
                ref_Q, ref_R = np.linalg.qr(reference.T)
                P = ref_Q @ ref_Q.T
                eeg_augment_project = eeg_augment @ P
                eeg_augment_final = np.concatenate((eeg_augment, eeg_augment_project), axis=1)
                weight = self.weights[:,:,fb_ii]
                template = self.templates[:,:,class_ii, fb_ii]
                signal = np.reshape(weight.T@eeg_augment_final, -1)
                templ = np.reshape(weight.T@template, -1)
                tmp = np.corrcoef(signal, templ)
                rho_mat[class_ii, fb_ii] = tmp[0, 1]

        rho_vec = rho_mat @ self.fb_coeffs # neither of Nakanishi & CM Wong use power, though TRCA2018 stated that \rho^2*weight. 
        predict = np.argmax(rho_vec)
        return predict, rho_vec


                                                 


    def update_weight(self, eeg_fb):
        # eeg_fb: tp ch blk fb class
        weights = np.zeros((self.channel_num*self.delay_num, self.subspace_num, self.fb_num))

        for fb_ii in range(self.fb_num):
            eeg = eeg_fb[:,:,:,fb_ii,:]
            if self.block_num == 1:
                raise ValueError("Check implementation in Matlab when Nb=1")
            weight = self._tdca(eeg)
            weights[:,:,fb_ii] = weight
        
        return weights

    
    def _tdca(self, eeg):
        # eeg: tp ch blk class
        _, channel_num, block_num, class_num = eeg.shape
        eeg_augment, _ = self.augment_eeg(eeg) # (ch*delay, tp*2, blk, class)
        similarity_between_class = np.zeros((self.delay_num*channel_num, self.delay_num*channel_num))
        similarity_within_class = np.zeros((self.delay_num*channel_num, self.delay_num*channel_num))

        for class_ii in range(class_num):
            for block_ii in range(block_num):
                eeg_augment_norm_1class = eeg_augment[:,:,block_ii,class_ii] - np.mean(eeg_augment[:,:,:,class_ii], -1)
                similarity_within_class += eeg_augment_norm_1class @ eeg_augment_norm_1class.T / block_num
                if block_num == 1:
                    raise ValueError("Check Implementation in Matlab when Nb=1")
            eeg_augment_norm_allclass = np.mean(eeg_augment[:,:,:,class_ii],-1) - np.mean(eeg_augment, (-2,-1))
            similarity_between_class += eeg_augment_norm_allclass @ eeg_augment_norm_allclass.T / class_num

        evalues, evectors = np.linalg.eig(np.linalg.inv(similarity_within_class) @ similarity_between_class)
        sort_descend_indices = np.argsort(evalues)[::-1]
        evalues = evalues[sort_descend_indices]
        weight_mat = evectors[:, sort_descend_indices[:self.subspace_num]]
        return weight_mat



    def update_template(self, eeg_fb):
        # eeg_fb: tp ch blk fb class
        templates = np.zeros((self.channel_num*self.delay_num, self.timepoint_num*2, self.class_num, self.fb_num))

        for fb_ii in range(self.fb_num):
            eeg_tmp = eeg_fb[:, :, :, fb_ii, :]
            if self.block_num == 1:
                raise ValueError("Check TDCA in Matlab for how to implement")
            _, eeg_template_augment = self.augment_eeg(eeg_tmp)
            templates[:, :, :, fb_ii] = eeg_template_augment
        return templates
    

    def augment_eeg(self, eeg, is_center=False):
        """
        Input:
            eeg - ndarray - (tp ch blk class)
        Output:
            eeg_augment - ndarray - (tp*2, ch*delay, blk, class)
        """
        _, channel_num, block_num, class_num = eeg.shape
        eeg_augment = np.zeros((channel_num*self.delay_num, 2*self.timepoint_num, block_num, class_num))

        for class_ii in range(class_num):
            reference = self.references[class_ii, :, :] # 2Nh * tp
            [ref_Q, ref_R] = np.linalg.qr(reference.T)
            P = ref_Q @ ref_Q.T

            
            for block_ii in range(block_num):
                x_augment = np.array([])
                for delay_ii in range(self.delay_num):
                    x = eeg[delay_ii:,:,block_ii, class_ii].T # ch tp
                    x = (x - np.mean(x, -1))/np.std(x, -1) if is_center else x # broadcase?
                    x_zeropad = np.concatenate((x, np.zeros((channel_num, delay_ii))), axis=1)
                    x_augment = np.concatenate((x_augment, x_zeropad), axis=0) if x_augment.size else x_zeropad

                x_augment_project = x_augment @ P
                x_augment_project = (x_augment_project - np.mean(x_augment_project, -1))/np.std(x_augment_project, -1) if is_center else x_augment_project # broadcase?
                eeg_augment[:, :, block_ii, class_ii] = np.concatenate((x_augment, x_augment_project), axis=1)

        eeg_augment_templates = np.mean(eeg_augment, axis=-2)

        return eeg_augment, eeg_augment_templates

    

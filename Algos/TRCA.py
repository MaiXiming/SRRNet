import numpy as np
import scipy
"""
Input: x: - ndarray -(blocks, classes, fbs, channels, timepoints) - no need for FB-preproc
"""

class TRCA:
    def __init__(self, fb_num=5, fb_coeff_a=-1.25, fb_coeff_b=0.25):
        self.fb_num = fb_num
        self.fb_coeff_a = fb_coeff_a
        self.fb_coeff_b = fb_coeff_b
        # self.class_num = 40

        self.fb_coeffs = np.arange(1, self.fb_num+1)
        self.fb_coeffs = np.power(self.fb_coeffs, self.fb_coeff_a) + self.fb_coeff_b


    def train(self, eeg_fb):
        """
        Input:
            eeg_fb: (blocks, classes, fbs, channels, timepoints)
        """
        self.block_num, self.class_num, fb_num_input, self.channel_num, self.timepoint_num = eeg_fb.shape
        if self.fb_num > fb_num_input:
            raise ValueError("TRCA Error: (fb_num of input eeg) < assigned fb_num in classdef.")
        
        self.templates = self.update_template(eeg_fb)
        self.weights = self.update_weight(eeg_fb)

        a = 1


    def get_weights(self):
        return self.weights
    

    def get_templates(self):
        return self.templates

    def detect(self, eeg_fb, is_ensemble=True):
        """
        Input:
            eeg - ndarray - (fbs, channels, timepoints)
        """
        rho_mat = np.zeros((self.class_num, self.fb_num))
        for fb_ii in range(self.fb_num):
            for class_ii in range(self.class_num):
                template = np.squeeze(self.templates[class_ii,fb_ii,:,])
                eeg = np.squeeze(eeg_fb[fb_ii,:,:])
                weight = np.squeeze(self.weights[:, fb_ii, :]) if is_ensemble \
                                    else np.squeeze(self.weights[:, fb_ii, class_ii])
                signal = np.reshape(np.transpose(eeg)@weight, -1)
                templ = np.reshape(np.transpose(template)@weight, -1)
                rho = np.corrcoef(signal, templ) # (250, 40), (250, 40) return 500*500: diff from Matlab, using vectors by reshape 
                rho_mat[class_ii,fb_ii] = rho[0,1]
        # rho_vec = np.power(rho_mat, 2) @ self.fb_coeffs 
        rho_vec = rho_mat @ self.fb_coeffs # neither of Nakanishi & CM Wong use power, though TRCA2018 stated that \rho^2*weight. 
        predict = np.argmax(rho_vec)
        return predict, rho_vec



    def update_weight(self, eeg_fb):
        weights = np.zeros((self.channel_num, self.fb_num, self.class_num))
        block_num = eeg_fb.shape[0]
        for fbii in range(self.fb_num):
            for classii in range(self.class_num):
                eeg = np.squeeze(eeg_fb[:, classii,fbii, :, :])
                eeg = np.expand_dims(eeg, 0) if block_num==1 else eeg
                weight = self._trca(eeg) # eig returns diff from Matlab
                weights[:, fbii, classii] = weight
        
        return weights
    

    def _trca(self, eeg):
        """
        Input: eeg - ndarray (blocks, channels, timepoints)
        """
        block_num, channel_num, _ = eeg.shape
        eeg = np.transpose(eeg, [1, 2, 0]) # chnl tp blk
        S = np.zeros((channel_num, channel_num))
        for block_ii in range(block_num-1):
            eeg_x1 = np.squeeze(eeg[:,:,block_ii])
            eeg_x1_mean = np.expand_dims(np.mean(eeg_x1, 1), 1)
            eeg_x1_mean = np.tile(eeg_x1_mean, (1, eeg_x1.shape[1]))
            eeg_x1 = eeg_x1 - eeg_x1_mean # mean across tp
            for block_jj in range(block_ii+1, block_num):
                eeg_x2 = np.squeeze(eeg[:,:,block_jj])
                eeg_x2_mean = np.expand_dims(np.mean(eeg_x2, 1), 1)
                eeg_x2_mean = np.tile(eeg_x2_mean, (1, eeg_x2.shape[1]))
                eeg_x2 = eeg_x2 - eeg_x2_mean # mean across tp
                S = S + np.matmul(eeg_x1, np.transpose(eeg_x2)) + np.matmul(eeg_x2, np.transpose(eeg_x1))
        
        eeg_lineup = np.reshape(eeg, (channel_num, -1))
        eeg_lineup_mean = np.expand_dims(np.mean(eeg_lineup, 1), 1)
        eeg_lineup_mean = np.tile(eeg_lineup_mean, (1, eeg_lineup.shape[1]))
        eeg_lineup = eeg_lineup - eeg_lineup_mean
        Q = np.matmul(eeg_lineup, np.transpose(eeg_lineup))
        _, eigen_vectors = scipy.linalg.eig(S, Q) # value different from Matlab
        return eigen_vectors[:, 0]


    def update_template(self, eeg_fb):
        templates = np.mean(eeg_fb, 0)
        return templates
        
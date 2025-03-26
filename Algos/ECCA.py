import numpy as np
import scipy
from sklearn.cross_decomposition import CCA

from PrepareData.References import SSVEPReference

"""
Ref: 2015 PNAS Xiaogang Chen "High speed SSVEP speller"
"""


class ECCA:
    def __init__(self, eeg_templates, timepoint_num, Nh=5, freqs=[8, 16], phases=[0, np.pi*3/2], fs=250, ):
        ## Input: tmpl (cls, Nch, tp)
        obj_ref = SSVEPReference(timepoint_num, Nh, freqs, fs, phases)
        self.references = obj_ref.get_refer_allfreqs() # (Nc, 2Nh, tp)

        self.templates = eeg_templates # cls Nch tp


    def detect(self, eeg):
        ## input: (channel, tp)
        eeg = np.transpose(eeg)
        cls_num = self.templates.shape[0]
        rhos = np.zeros((cls_num,))
        for cii in range(cls_num):
            ref = np.transpose(self.references[cii,:,:])
            tmpl = np.transpose(self.templates[cii,:,:])

            ## Standard CCA <test, templates>
            r1, _, _ = self.CCA(eeg, ref)

            ## Weights
            _, w2, _ = self.CCA(eeg, tmpl)
            _, w3, _ = self.CCA(eeg, ref)
            _, w4, _ = self.CCA(tmpl, ref)
            _, w51, w52 = self.CCA(eeg, tmpl)
            
            r2 = np.corrcoef((eeg@w2).T, (tmpl@w2).T)[0, 1]
            r3 = np.corrcoef((eeg@w3).T, (tmpl@w3).T)[0, 1]
            r4 = np.corrcoef((eeg@w4).T, (tmpl@w4).T)[0, 1]
            r5 = np.corrcoef((tmpl@w51).T, (tmpl@w52).T)[0, 1]

            tmp = np.array([r1, r2, r3, r4, r5])
            rho = np.sum(np.sign(tmp) * tmp**2)

            rhos[cii] = rho
        
        predict = np.argmax(rhos)
        return predict, rhos

            

    def CCA(self,x, y):
        cca = CCA(n_components=1)
        cca.fit(x, y)
        xc, yc = cca.transform(x, y)
        rho = np.corrcoef(xc.T, yc.T)
        A = cca.x_weights_
        B = cca.y_weights_
        return rho[0,1], A, B


if __name__ == '__main__':
    ## ECCA
    freqs = np.linspace(8, 15.8, 40)
    phases = np.linspace(0, np.pi, 40)
    tp = 250
    fs = 250

    eeg_templates = np.random.rand(40, 9, tp)
    eeg_test = np.random.rand(9, tp)

    ecca = ECCA(eeg_templates, tp, 5, freqs, phases, fs)
    print(ecca.detect(eeg_test))

    # ## CCA
    # x = np.array([[1,2], [3,4], [5,6]])
    # y = np.array([[4,3], [2,1], [6,5]])
    # cca = CCA(n_components=2)
    # cca.fit(x, y)

    # A = cca.x_weights_  # Coefficients for X
    # B = cca.y_weights_  # Coefficients for Y
    # print(A)
    # print(B)    

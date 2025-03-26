from PrepareData.load_dataset_parameters import *
from PrepareData.References import *

from scipy import signal

class SSVEPLoader:
    """
    Load SSVEP dataset and preprocess (filtering).
    Inputs:
        args: look at 1fold.py for details
    Returns:
        object

    Use:
        # Use args to control parms, e.g., dataset, sliding, etc 
        ssvepdata = SSVEPLoader(args)
        args = ssvepdata.update_args()
        ## Get data
        eeg_all = ssvepdata.get_eeg_data() # (block slide class chnl tp)
        sc_template = ssvepdata.get_sc_template() # (slide class 2Nh tp)
    """
    def __init__(self, args):
        """Load data, extract segments (sliding in each trial), and preprocess-filtering"""
        self.args = args
        self.parameters = load_dataset_parameters(self.args.dataset, self.args.server) # bug? need sorted frequency? 8-16, now it is 8-16 8.2-15.2 ...

        ## Filter
        # Wp = [6, 90]
        # N = 4
        # self.filter_b, self.filter_a = signal.butter(N, Wp, btype='bandpass', fs=self.parameters['sampling_rate']) # bug: cheb?
        rfs = self.parameters['sampling_rate'] / 2
        wp, ws = np.array([6, 90]), np.array([4, 100])
        Wp, Ws = wp / rfs, ws / rfs
        N, Wn = signal.cheb1ord(Wp, Ws, 3, 40)
        self.filter_b, self.filter_a = signal.cheby1(N, 0.5, Wn, 'bandpass')

        if 1e-3 < self.args.step <= 1: # second
            self.step_timepoint = int(self.args.step * self.parameters['sampling_rate'])
        elif self.args.step > 1: # win%
            self.step_timepoint = int(self.args.step * 0.01 * self.args.window * self.parameters['sampling_rate'])
        else:
            raise ValueError("args.step value invalid")


        # ## Load eeg whole trial (blk, class, chnl, tp)
        # self.data_trial = self.load_eeg_trial_subjs(self.args.dataset, self.args.subject) # block class chnl tp
        # self.sc_template_trial = SSVEPReference(timepoint_num=self.parameters['trial_timepoint'], harmonic_num=args.harmonic_num, frequencies=self.parameters['frequencies'], sampling_rate=self.parameters['sampling_rate'], phases=self.parameters['phases']).get_refer_allfreqs() # class 2Nh tp # bug: delay? align?

        # ## Slide into epochs in each trial (blk, slides, class, chnl, tp)
        # self.data_epoch, self.sc_template = self.slide_into_epoch(self.data_trial, self.sc_template_trial) # (block slide class chnl tp) & (slide class 2Nh tp)
        # self.data_epoch = self.preprocessing(self.data_epoch) if self.args.is_filter else self.data_epoch
        # self.slide_num = self.data_epoch.shape[1]

    def get_sc_template(self):
        """
        Returns:
            sine-cosine template - ndarray (slide, class, 2Nh, tp): sine-cosine template of all freqs 
        """
        sc_template_trial = SSVEPReference(timepoint_num=self.parameters['trial_timepoint'], harmonic_num=self.args.harmonic_num, frequencies=self.parameters['frequencies'], sampling_rate=self.parameters['sampling_rate'], phases=self.parameters['phases'], is_phase_harmonic=self.args.is_phase_harmonic).get_refer_allfreqs() # class 2Nh tp # bug: delay? align?
        sc_template_epoch = self.slide_trial2epoch_sc_template(sc_template_trial)
        return sc_template_epoch
        # return self.sc_template
    
    def get_eeg_data(self, subjs):
        """
        Returns:
            data_epoch (ndarray, (block slide class chnl tp)): filtered EEG epochs 
        """
        data_trial = self.load_eeg_trial_subjs(self.args.dataset, subjs) # block class chnl tp
        data_epoch = self.slide_trial2epoch_data(data_trial)
        data_epoch = self.preprocessing(data_epoch) if self.args.is_filter else data_epoch
        return data_epoch
    
    # def get_eeg_data(self):
    #     """
    #     Returns:
    #         data_epoch (ndarray, (block slide class chnl tp)): filtered EEG epochs 
    #     """
    #     return self.data_epoch


    

    # def get_sc_template(self):
    #     """
    #     Returns:
    #         sine-cosine template - ndarray (slide, class, 2Nh, tp): sine-cosine template of all freqs 
    #     """
    #     return self.sc_template


    def preprocessing(self, data):
        eeg_preprocess = data
        eeg_preprocess = signal.detrend(eeg_preprocess, axis=-1) if self.args.is_detrend else eeg_preprocess
        eeg_preprocess = signal.filtfilt(self.filter_b, self.filter_a, eeg_preprocess, 
                            axis=-1, padlen=3*(max(len(self.filter_b),len(self.filter_a))-1))
        return eeg_preprocess

    def slide_trial2epoch_data(self, data_raw):
        # step_timepoint = int(self.args.step * self.parameters['sampling_rate'])
        step_timepoint = self.step_timepoint
        timepoint_num = int(self.args.window * self.parameters['sampling_rate'])
        data_epoch = []
        index_bgn = 0
        while (index_bgn + timepoint_num) <= data_raw.shape[-1]:
            data_epoch.append(data_raw[:,:,:,(index_bgn):(index_bgn+timepoint_num)])
            # sc_template.append(sc_template_raw[:,:,(index_bgn):(index_bgn+timepoint_num)])

            index_bgn += step_timepoint
        
        data_epoch = np.stack(data_epoch, axis=1) # block slide class chnl tp
        # sc_template = np.stack(sc_template, axis=0) # slide class chnl tp

        return data_epoch

    
    def slide_trial2epoch_sc_template(self, sc_template_raw):
        # step_timepoint = int(self.args.step * self.parameters['sampling_rate'])
        step_timepoint = self.step_timepoint
        timepoint_num = int(self.args.window * self.parameters['sampling_rate'])
        sc_template = []
        index_bgn = 0
        while (index_bgn + timepoint_num) <= sc_template_raw.shape[-1]:
            # data_epoch.append(data_raw[:,:,:,(index_bgn):(index_bgn+timepoint_num)])
            sc_template.append(sc_template_raw[:,:,(index_bgn):(index_bgn+timepoint_num)])

            index_bgn += step_timepoint
        
        # data_epoch = np.stack(data_epoch, axis=1) # block slide class chnl tp
        sc_template = np.stack(sc_template, axis=0) # slide class chnl tp

        return sc_template
    
    def slide_into_epoch(self, data_raw, sc_template_raw):
        """
        Slide window strategy to extract multi epochs in one trial.
        Note: also slide sine-cosine template because phase of mean template matters for template matching decoding, and we use sine-cosine template to regress mean template. Since sample size is too small without sliding (1 for each seen class), we need to slide. thus we need to slide sine-cosine template also.
        
        Parameters
        ----------
        data_raw - ndarray (block class chnl tp): data for each trial
        sc_template_raw - ndarray (class 2Nh tp): sine-cosine template of all classes.

        Returns
        ----------
        data_epoch - ndarray (block slide class chnl tp)
        sc_template - ndarray (slide class chnl tp)
        """

        # step_timepoint = int(self.args.step * self.parameters['sampling_rate'])
        step_timepoint = self.step_timepoint
        timepoint_num = int(self.args.window * self.parameters['sampling_rate'])
        data_epoch, sc_template = [], []
        index_bgn = 0
        while (index_bgn + timepoint_num) <= data_raw.shape[-1]:
            data_epoch.append(data_raw[:,:,:,(index_bgn):(index_bgn+timepoint_num)])
            sc_template.append(sc_template_raw[:,:,(index_bgn):(index_bgn+timepoint_num)])

            index_bgn += step_timepoint
        
        data_epoch = np.stack(data_epoch, axis=1) # block slide class chnl tp
        sc_template = np.stack(sc_template, axis=0) # slide class chnl tp

        return data_epoch, sc_template


    def load_eeg_trial_subjs(self, dataset, subject):
        """Load raw eeg from .dat file.
        Parameters
        ----------
        dataset - str: benchmark or beta
        subject - int/list: subject No

        Returns
        ----------
        data_raw - ndarray (block,class,chnl,tp): raw EEG data. tp=timepoint=whole trial (excluding cue+rest)
        """
        if isinstance(subject, int):
            data = self.load_eeg(dataset, subject)
        else: # list
            data = []
            for ss in subject:
                data.append(self.load_eeg(dataset, ss))
            data = np.stack(data, axis=0) # subject, block, class, chnl, tp
            data = np.reshape(data, (-1, data.shape[2], data.shape[3], data.shape[4])) # subject*block, class, chnl, tp

        data_raw = self.extract_wholetrial(data)
        return data_raw # block, class, chnl, tp


    def load_eeg(self, dataset, subject):
        """
        Load eeg from raw files.
        Parameters
        ----------
        dataset - str: public dataset name
        subject - int: subject No
        Returns
        ----------
        eeg_raw - ndarray (block,class,chnl,tp): raw eeg data. (including cue + rest)

        """
        suffix = 's' if (self.args.dataset == 'ucsd') else 'S'
        # path_data = os.path.join(self.parameters['path'], f"s{subject + 1}")
        path_data = os.path.join(self.parameters['path'], f"{suffix}{subject + 1}")
        rawdata = scipy.io.loadmat(path_data)
        if dataset == 'ucsd':
            eeg_raw = rawdata['eeg'] # targets * chnls * smpls * trainsmpls
            eeg_raw = np.transpose(eeg_raw, (3, 0, 1, 2)) # block class chnl tp
        elif dataset == 'benchmark':
            eeg_raw = rawdata['data'] # chnl smpl classes trainsmpl
            eeg_raw = np.transpose(eeg_raw, (3, 2, 0, 1))
        elif dataset == 'beta':
            eeg_raw = rawdata['data'][0][0]['EEG'] # chnl smpl trainsmpl classes 
            eeg_raw = np.transpose(eeg_raw, (2, 3, 0, 1))
            eeg_raw = eeg_raw[:,:,:,:750] ## bug: s15-s69 has 3s stimulation, ignore for now
        else:
            raise ValueError("self.args.dataset not found!\n")
        print(f"\r load EEG data S{subject}", end='')
        
        return eeg_raw # block class chnl tp (cue+trial+rest)
    

    def extract_wholetrial(self, eeg_raw):
        """
        Extract task segment from each trial (consider delay=0.13/0.14s)
        Parameters
        ----------
        eeg_raw - ndarray,(block class chnl tp)
        
        Returns
        ----------
        eeg_interest - ndarray,(block class chnl tp), excluding cue + rest
        """
        timepoint_begin_in_trial = np.round(self.parameters['onset_timepoint'] + self.parameters['delay_timepoint'])
        timepoint_end_in_trial = np.round(timepoint_begin_in_trial + self.parameters['trial_second'] * self.parameters['sampling_rate'])
        timepoint_indices_in_trial = np.arange(timepoint_begin_in_trial, timepoint_end_in_trial).astype(int)
        # bug? sequentially slice. together cause Error. why?
        eeg_interest = eeg_raw[:, :, self.parameters['channels_decode_indices'], :]
        eeg_interest = eeg_interest[:, :, :, timepoint_indices_in_trial]

        return eeg_interest # block class chnl tp (trial)

    
    def update_args(self):
        self.args.class_num = self.parameters['class_num']
        self.args.block_num = self.parameters['block_num']
        self.args.chnl_num = self.parameters['channels_decode_indices'].shape[0]
        self.args.timepoint_num = np.round(self.parameters['sampling_rate'] * self.args.window).astype(int)
        self.args.input_shape = (1, self.args.chnl_num, self.args.timepoint_num)
        self.args.sampling_rate = self.parameters['sampling_rate']
        self.args.block_num = self.parameters['block_num']
        self.args.frequencies = self.parameters['frequencies']
        self.args.phases = self.parameters['phases']
        self.args.harmonic_x2 = self.args.harmonic_num * 2

        return self.args

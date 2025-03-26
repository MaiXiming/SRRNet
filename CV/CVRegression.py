import time
import pickle
from torch.utils.data import DataLoader, TensorDataset

from CV.CrossValidation import *
from plots import *

class CVRegression(CrossValidation):
    """
    Leave-one-block-out (lobo) cross validation on Sine2SSVEP regression.
    Main idea: Filterbank{regress sine2ssvep, TRCA spatially filtered, then flatten and correlation}
    """

    def __init__(self, args):
        super(CVRegression, self).__init__(args)

    def lobo_fb(self, eeg_1subj, sc_template, block_test, eeg_subjs_pretrain=None):
        """
        Perform regress/filter/correlation on each sub-band, then combine using n**-1.25+0.25
        Inputs: same as lobo

        Returns: metrics dict, including
            acc: accuracy for this block
            predicts: prediction labels on Nclass test trials (for confusion matrix computing)
            others
        """
        filterbank = Filterbank(filterbank_num=self.args.fb_num, sampling_rate=self.args.sampling_rate)
        rhos_fbs = []
        losses, corrs = [], []
        for fbii in range(self.args.fb_num):
            self.args.fbii = fbii
            eeg_fb = filterbank.filter_subband(eeg_1subj, fbii)
            eeg_subjs_pretrain_fb = filterbank.filter_subband(eeg_subjs_pretrain, fbii) if self.args.is_pretrain else []
            metrics_fbii, rhos_fbii = self.lobo_regress(eeg_fb, sc_template, block_test, eeg_subjs_pretrain_fb, fbii)
            rhos_fbs.append(rhos_fbii)

            losses.append([metrics_fbii['loss_train'], metrics_fbii['loss_valid'], metrics_fbii['loss_unseen']])
            corrs.append([metrics_fbii['corr_train'], metrics_fbii['corr_valid'], metrics_fbii['corr_unseen']])

        ## Combine FB
        rhos_fbs = np.stack(rhos_fbs, axis=-1) # (test class fb)
        rho_vec = np.matmul(rhos_fbs, filterbank.fb_weights) # (test class)
        predicts = np.argmax(rho_vec, axis=1)
        acc = np.mean(predicts==np.array(self.args.labels))
        print(f'Subject={self.args.subject}, block_test={block_test}, Acc={acc:.4f}')

        losses_mean = np.mean(np.array(losses), axis=0)
        corrs_mean = np.mean(np.array(corrs), axis=0)
        metrics = {'loss_train': losses_mean[0], 'loss_valid': losses_mean[1], 'loss_unseen': losses_mean[2],
                   'corr_train': corrs_mean[0], 'corr_valid': corrs_mean[1], 'corr_unseen': corrs_mean[2],
                   'acc': acc, 'predicts': predicts}
        return metrics

    
    def lobo_regress(self, eeg_1subj, sc_template, block_test, eeg_subjs_pretrain, fbii=0):
        """
        Lobo: Note that for regression, (N-1) training blocks is further separated into seen / unseen, the unseen part shouldn't used for model training, only for analysis&visualization. Only the seen part in training blocks is the true training set, which could be used for model training.
        Process includes: 
            - dataset separation: data=train_allclass&test, train_allclass=seen&unseen, seen=train&valid 
            - train regression model, 
            - predict mean template of unseen, 
            - calculate spatial filters, 
            - predict

        Inputs:
            eeg_1subj - ndarray (block,slide,class,chnl,tp): slided eeg epochs of the whole dataset
            sc_template - ndarray (slide,class,N2h,tp): slided sine-cosine template, phase aligned with each eeg epoch
            block_test - list: blocks for testing (1 in lobo case)
            args - structure: arguements set in `onefold.py`

        Returns
            metrics: including acc/predicts/...
            rhos - ndarray (test samples/Nclass, Nclass): correlation to each class for each sample (each row)
        """
        self.args.fbii, self.args.blkii = fbii, block_test
        ## Prepare training set
        reg_trainset, reg_validset, unseenset, testblocks, seenset = self.separate_data_and_normalize(eeg_1subj, sc_template, block_test) # seen=train+valid
        pretrainset = self.extract_seen_pretrain_allblocks(eeg_subjs_pretrain, sc_template) if self.args.is_pretrain==1 else None

        ## Regress templates using DL model
        model, best_dict = self.setup_and_train_reg(reg_trainset, reg_validset, pretrainset=None)
        corr_unseen, loss_unseen = 0, 0 ## evaluate_model_on_unseen(model, unseenset, self.args, loss='mseloss')
        
        ssvep_templates_combine, ssvep_templates_regress = self.get_ssvep_template_meanseen_regunseen(reg_trainset, reg_validset, model, sc_template[0,:,:,:]) # cls subspace tp
        


        ## Save templates
        # true_unseen_templates = np.mean(unseenset['data'][:,0,:,:,:], 0) # unseen*chnl*tp
        # recon_unseen_templates = ssvep_templates_combine[unseenset['label']]
        # file = open(self.args.fn_detail, 'rb')
        # data = pickle.load(file)
        # data['true_templates'][:, fbii, :, :] = true_unseen_templates
        # data['recon_templates'][:, fbii, :, :] = recon_unseen_templates
        # file.close()

        # file = open(self.args.fn_detail, 'wb')
        # pickle.dump(data, file)
        # file.close()
        update_template_in_detail(self.args, seenset, unseenset, ssvep_templates_regress, fbii)



        ## Spatial filters
        data_seen = np.concatenate((reg_trainset['data'][:,0,:,:,:], reg_validset['data'][:,0,:,:,:]), axis=1) # along class dim --> (block,class,chnl,tp)
        weights_sf_seen = self.calculate_spatialfilters(data_seen, model=self.args.spatialfilter)
        
        ## Predict
        data_test = np.squeeze(testblocks['data']) # class chnl tp
        predicts, rhos = self.decode_testdata(data_test, ssvep_templates_combine, weights_sf_seen)
        
        acc = np.mean(predicts==np.array(testblocks['label']))
        print(f'Subject={self.args.subject}, block_test={block_test}, fbii={fbii}, Acc={acc:.4f}')

        metrics = {'loss_train': best_dict['loss_train'], 'loss_valid': best_dict['loss_valid'], 'loss_unseen': loss_unseen,
                   'corr_train': best_dict['corr_train'], 'corr_valid': best_dict['corr_valid'], 'corr_unseen': corr_unseen,
                   'acc': acc, 'predicts': predicts}
        

        ## For Offline Analysis
        if self.args.is_plot_template == 1:
            explore_template(ssvep_templates_regress, reg_trainset, reg_validset, unseenset, self.args)

        if self.args.is_plot_weights == 1:
            plot_weights(best_dict, self.args)

        if self.args.is_tsne_data == 1:
            data_trca = np.matmul(weights_sf_seen, data_test)
            outputs = {'raw': data_test,
                       'trca': data_trca,
                       'predict': rhos} # (sample, class)
            with open(f"Analysis/tSNE/data/{self.args.dataset}-u{self.args.unseen_num}-t{self.args.window}-{self.args.spatialfilter}-{self.args.model}-fb1-s{self.args.subject}-b{self.args.blkii}.pickle", 'wb') as file:
                pickle.dump(outputs, file)
        
        return metrics, rhos
    
    
    def decode_testdata(self, data_test, ssvep_templates, weights_sf_seen):
        """
        Predict labels for test trials
        Inputs:
            data_test - ndarray (samples/Nclass, chnl, tp)
            ssvep_templates - ndarray (Nclass, chnl, tp)
            weights_sf_seen - ndarray (subspace/Nseen, chnl)

        Returns:
            predicts - list: predicted labels
            rhos_mat - ndarray (samples, Nclass): correlation betw each test sample and templates of each class
        """
        ## eCCA: Directly decode without filters
        # if self.args.spatialfilter == 'ecca':
        #     ecca = ECCA(ssvep_templates, int(self.args.window*self.args.sampling_rate), self.args.harmonic_num, self.args.frequencies, self.args.phases, self.args.sampling_rate)
            
        #     sample_num = data_test.shape[0]
        #     predicts = []
        #     rhos_mat = np.zeros((sample_num, self.args.class_num))
        #     for sample_i in range(sample_num):
        #         test_epoch = data_test[sample_i]
        #         # predict, rho_vec = self.detect_flatten_and_corr(test_epoch, ssvep_templates)
        #         predict, rho_vec = ecca.detect(test_epoch)
        #         predicts.append(predict)
        #         rhos_mat[sample_i, :] = rho_vec
        #     return predicts, rhos_mat

        if self.args.spatialfilter != 'ecca':
            ## Apply spatial filters
            data_test = np.matmul(weights_sf_seen, data_test)
            ssvep_templates = np.matmul(weights_sf_seen, ssvep_templates)
        else:
            ## eCCA: Directly decode without filters
            ecca = ECCA(ssvep_templates, int(self.args.window*self.args.sampling_rate), self.args.harmonic_num, self.args.frequencies, self.args.phases, self.args.sampling_rate)

        sample_num = data_test.shape[0]
        predicts = []
        rhos_mat = np.zeros((sample_num, self.args.class_num))
        for sample_i in range(sample_num):
            test_epoch = data_test[sample_i]
            if self.args.spatialfilter == 'ecca':
                predict, rho_vec = ecca.detect(test_epoch)
            else:
                predict, rho_vec = self.detect_flatten_and_corr(test_epoch, ssvep_templates)
            predicts.append(predict)
            rhos_mat[sample_i, :] = rho_vec
        return predicts, rhos_mat
    

    def detect_flatten_and_corr(self, eeg, ssvep_templates):
        """
        Flatten into 1d vector and do correlations.
        Inputs:
            eeg: spatial filtered test eeg sample
            ssvep_templates: spatial filtered templates
        
        Returns:
            predict - int: predicted label (argmax correlation)
            rho_vec - list: correlation <test eeg, templates of all classes>
        """
        class_num, _, _ = ssvep_templates.shape
        rho_vec = np.zeros((class_num))
        for class_ii in range(class_num):
            template = np.squeeze(ssvep_templates[class_ii,:,:])
            signal = np.reshape(np.transpose(eeg), -1)
            templ = np.reshape(np.transpose(template), (-1))
            rho = np.corrcoef(signal, templ)
            rho_vec[class_ii] = rho[0,1]
        predict = np.argmax(rho_vec)
        return predict, rho_vec


    def get_ssvep_template_meanseen_regunseen(self, trainset, validset, model, sc_template_s0):
        ## Regress ssvep template for unseen classes using model&sc_template, mean() ssvep template for seen classes using trainset&validset
        ssvep_template_regress = self.regress_ssvep_template(model, sc_template_s0)
        ## Integrate mean & regressed templates into final ssvep templates
        ssvep_templates_output = np.zeros_like(ssvep_template_regress)
        ssvep_templates_output[:] = ssvep_template_regress[:]
        if self.args.is_tmpl_trueseen == 1:
            ## Don't use data_seen since order is wrong, ok for spatial filtering, not ok for templates
            ssvep_templates_output[self.args.label_train,:,:] = np.mean(trainset['data'][:,0,:,:,:], axis=0)
            ssvep_templates_output[self.args.label_valid,:,:] = np.mean(validset['data'][:,0,:,:,:], axis=0)
        return ssvep_templates_output, ssvep_template_regress


    def regress_ssvep_template(self, model, sc_template_s0):
        sc_template_s0 = torch.from_numpy(sc_template_s0.astype('float32')).float().to(self.args.device)
        sc_template_s0 = torch.unsqueeze(sc_template_s0, dim=1)
        ssvep_template_regress = model(sc_template_s0)
        ssvep_template_regress = ssvep_template_regress.detach().cpu().numpy() # !!! seen class also from regression
        return ssvep_template_regress
 

    def extract_seen_pretrain_allblocks(self, eeg_subjs, sc_template):
        # trainblocks_pretrain = {'data': eeg_subjs} # (N-1)subjects*blocks, all classes
        seenset, _ = self.separate_into_seen_unseen(eeg_subjs, sc_template) ## seen classes, no valid
        if self.args.is_normalize == 1:
            m, s = np.mean(seenset['data']), np.std(seenset['data'])
            seenset = self.normalize_eeg(m, s, seenset)
        return seenset

    
    
    def separate_data_and_normalize(self, eeg, sc_template, block_test):
        """
        Separate entire dataset and normalize.
            dataset = train blocks with all classes + test block
                train blocks with all classes = seen + unseen
                    seen = train + validate

        Inputs:
            eeg - ndarray,(block, slide, class, chnl, tp): eeg epochs of entire dataset
            sc_template - ndarray,(slide, class, chnl, tp): sine-cosine template epochs of all freqs, aligned with eeg epochs
            block_test - list: 1 for lobo

        Returns
            trainset/validset/unseenset/seenset - dictionary: including data, sc_template, label
            testblocks - dictionary: including data, label, block
        """
        trainallclass_blocks, testblocks = self.separate_all_into_train_test(eeg, block_test, self.args.blocks)
        seenset, unseenset = self.separate_into_seen_unseen(trainallclass_blocks['data'], sc_template)
        trainset, validset = self.separate_into_train_valid(seenset)

        if self.args.is_normalize == 1:
            ## Careful with data leakage: don't use unseen/testblocks
            norm_mean, norm_std = self.get_norm_params(trainset, validset)
            trainset = self.normalize_eeg(norm_mean, norm_std, trainset)
            validset = self.normalize_eeg(norm_mean, norm_std, validset)
            seenset = self.normalize_eeg(norm_mean, norm_std, seenset)
            unseenset = self.normalize_eeg(norm_mean, norm_std, unseenset)
            testblocks = self.normalize_eeg(norm_mean, norm_std, testblocks)

        return trainset, validset, unseenset, testblocks, seenset
    

    def setup_and_train_reg(self, trainset, validset, pretrainset):
        """
        numpy2torch data and train model.
        Inputs:
            trainset/validset - dict: including sine template and SSVEP templates (mean)
            pretrainset - dict: unused here
        Returns:
            model: train model
            best_dict: for recording loss&corr in training stage
            (side effect: save model parms in `./checkpoints`
        """
        ## Prepare training set
        train_loader, X_train, y_train, freqs_train = self.np2torch_reg(trainset)
        _, X_valid, y_valid, freqs_valid = self.np2torch_reg(validset)
        if pretrainset:
            pretrain_loader, X_pretrain, y_pretrain, freqs_pretrain = self.np2torch_reg_pretrain(pretrainset)
        
        ## Select model, loss, opt
        model = select_model(self.args, y_train.shape[-2])
        model.apply(init_weights)
        criterion = get_criterion(self.args.loss)
        
        parms_model = model.parameters()
        optimizer = get_optimizer(parms_model, self.args, lr=self.args.lr)
        best_dict = self.train_reg(model, optimizer, criterion, train_loader, X_train, y_train, freqs_train, X_valid, y_valid, freqs_valid)

        ## Save best model   
        torch.save(best_dict, self.args.save_path) # save once at the end of training is enough
        print(f"\r***Best score*** Epoch {best_dict['epoch']+1:03d}") # best loss not considered

        # Load best model
        del model
        model = select_model(self.args, y_train.shape[-2])
        model.load_state_dict(best_dict['model_state_dict'])
        print(f"Load model state on epoch {best_dict['epoch']+1}")

        # ## Visualization
        # visualize_weight_harmonic(model, self.args)
        return model, best_dict

    def train_reg(self, model, optimizer, criterion, train_loader, X_train, y_train, freqs_train, X_valid, y_valid, freqs_valid):
        ## Gradient descent for epochs
        print('Start training!')
        print("Model size: ", get_trainable_parameter_num(model))
        print("Training set data size: ", X_train.shape)
        trainsample_num = X_train.shape[0]
        best_score = 1e4 # ensure epoch1 will save
        best_epoch = 0
        losses_train, losses_valid, corrs_train, corrs_valid = [],[],[],[]
        for epoch in range(self.args.epoch_num):
            t1 = time.time()
            ## Train
            model.train()
            loss_epoch = 0
            ## Mini batch optim
            for batch_ii, batch_data in enumerate(train_loader):
                x_batch, y_batch, freqs_batch = batch_data
                if self.args.model == 'streggroup':
                    out = model(x_batch, freqs_batch)
                else:
                    out = model(x_batch)
                loss = criterion(out, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.data

            ## Evaluate on trainset, validset --> loss, metrics=correlation
            corr_train, loss_train = self.evaluate_performance(model, criterion, X_train, y_train)
            corr_valid, loss_valid = self.evaluate_performance(model, criterion, X_valid, y_valid)
            losses_train.append(loss_train)
            losses_valid.append(loss_valid)
            corrs_train.append(corr_train)
            corrs_valid.append(corr_valid)

            ## Print info
            dt = time.time() - t1
            eff = trainsample_num / dt
            print(f'\rEpoch {epoch+1}, Train loss: {loss_train:.4f}, Valid loss: {loss_valid:.4f}; ' + 
                f'Train corr: {corr_train:.4f}, Valid corr: {corr_valid:.4f}', end='')
            str1 = f'\rEpoch {epoch+1}, LOSS train={loss_train:.4f}, valid={loss_valid:.4f}; CORR train={corr_train:.4f}, valid={corr_valid:.4f}'
            str2 = f'(Time={dt:.2f}, {eff:.2f} samples/sec)'
            print(str1+str2, end='')

            ## Plot
            is_plot = (epoch+1)%30==0 or epoch == (self.args.epoch_num-1)
            if self.args.is_losscurve and is_plot:
                curves = {'loss_train': losses_train, 'loss_valid': losses_valid,
                        'corr_train': corrs_train, 'corr_valid': corrs_valid}
                # plot_training_curve([losses_train, losses_valid, corrs_train, corrs_valid], epoch, args)
                plot_training_curve(curves, epoch, self.args.epoch_num, self.args)

            ## Save if beat the best
            current_score = loss_valid # corr_valid loss_valid
            is_earlystop = (self.args.is_earlystopping == 1)
            is_beat_best = (current_score < best_score)
            is_save = (not is_earlystop) or (is_earlystop and is_beat_best)
            if is_save:
                best_score, best_epoch, best_dict = self.update_best_dict(current_score, epoch, model, loss_train, loss_valid, corr_train, corr_valid)
            else:
                is_stop = (epoch - best_epoch) >= self.args.es_patience
                if is_stop:
                    break

        print()
        return best_dict

    def update_best_dict(self, current_score, epoch, model, loss_train, loss_valid, corr_train=0, corr_valid=0):
        best_score = current_score
        best_epoch = epoch
        best_dict = {'model_state_dict': model.state_dict(),
                    'epoch': best_epoch, 'best_score': best_score,
                    'loss_train': loss_train, 'corr_train': corr_train, 
                    'loss_valid': loss_valid, 'corr_valid': corr_valid, 
                    }
        return best_score, best_epoch, best_dict

    def evaluate_performance(self, model, criterion, X, y):
        """
        Evaluate model performance in terms of criterion on dataset (X,y)
        Parameters
        ----------
        model - class object: trianed model
        criterion - pytorch function: MSELoss, etc
        X - ndarray (samples,2Nh,tp): sine-cosine templates
        y - ndarray (samples, chnl, tp): corresponding mean templates ("labels" of sc_template)
        """
        model.eval()
        X = X.detach()
        y = y.detach()
        out = model(X)
        loss = criterion(out, y)
        corr_mean = cal_corr(out, y).detach().cpu().numpy() # bug? might take long time
        # corr_mean, corrs = 0, 0
        loss = loss.detach().cpu().numpy()
        return corr_mean, loss


    def np2torch_reg_pretrain(self, pretrainset):
        ## pretrainset['data']: (N-1)*blocks, slide, Nseen, chnl, tp; ['sc_template']: slide, Nseen, chnl, tp
        ## not mean all subjects, but to mean for each subjs
        subjs_pretrain = self.args.subjects - 1
        X_train, y_train, freqs = [], [], []
        for sii in range(subjs_pretrain):
            trainset = {'data': pretrainset['data'][(sii*self.args.block_num):((sii+1)*self.args.block_num),:,:,:,:],
                        'sc_template': pretrainset['sc_template'],
                        'label': pretrainset['label'],
            }
            _, x, y, f = self.np2torch_reg(trainset, self.args) # slide*Nseen, samples > 1 subj train/valid
            X_train.append(x)
            y_train.append(y)
            freqs.append(f)
        X_train = torch.stack(X_train, dim=0)
        y_train = torch.stack(y_train, dim=0)
        X_train = torch.reshape(X_train, (-1, 1, X_train.shape[-2], X_train.shape[-1]))
        y_train = torch.reshape(y_train, (-1, y_train.shape[-2], y_train.shape[-1]))
        freqs = torch.stack(freqs, dim=0)
        freqs = torch.reshape(freqs, (-1,))
        dataset_train = TensorDataset(X_train, y_train, freqs)
        train_loader = DataLoader(dataset_train, batch_size=self.args.batch_size, shuffle=True)

        return train_loader, X_train, y_train, freqs


    def np2torch_reg(self, trainset):
        ## Calculate mean template
        y_train = np.mean(trainset['data'], axis=0)

        freqs = [self.args.frequencies[trainset['label']] for _ in range(trainset['data'].shape[1])]
        freqs = np.stack(freqs, axis=0) # slide, Nseen

        ## Flatten into (samples, chnl, tp) samples=slide*class
        chnl_num, timepoint_num = trainset['data'].shape[-2], trainset['data'].shape[-1]
        harmonic_x2 = trainset['sc_template'].shape[-2]
        X_train = np.reshape(trainset['sc_template'], (-1, harmonic_x2, timepoint_num))
        # y_train = np.reshape(y_train, (-1, subspace_num, timepoint_num))
        y_train = np.reshape(y_train, (-1, chnl_num, timepoint_num))
        freqs = np.reshape(freqs, (-1))

        ## Train torch
        X_train = torch.from_numpy(X_train.astype('float32')).float().to(self.args.device)
        X_train = torch.unsqueeze(X_train, dim=1)
        y_train = torch.from_numpy(y_train.astype('float32')).float().to(self.args.device)
        freqs = torch.from_numpy(freqs.astype('float32')).float().to(self.args.device) # freqs, not label

        dataset_train = TensorDataset(X_train, y_train, freqs)
        train_loader = DataLoader(dataset_train, batch_size=self.args.batch_size, shuffle=True)

        return train_loader, X_train, y_train, freqs
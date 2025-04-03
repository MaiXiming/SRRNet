from CV.CrossValidation import *
from torch.utils.data import *
from utils import *

import time

"""
Unmentioned details in iGZSL paper: reason for choosing
- Pretrain data = (N-1) subjs, all blocks, seen classes, sliding 
    (confirmed with 1st author via email)
    providing sufficient data (all blocks) while not including test subj. Assuming that this paradigm only need seen classes for training, so pre-train subjects' data also not including unseen class.  
- finetune = test subj, seen classes, (N-2) blocks, sliding 
    (confirmed with 1st author via email)
- CNet = test subj, seen, (N-2) blocks, no sliding
    (confirmed with 1st author via email)
    as paper indicated.
- fixed training = test subj, seen classes, (N-1) blocks (train+valid1), no sliding
    (confirmed with 1st author via email)
    as paper indicated, while train+valid could provide more data
- Optimizer = AdamW, lr=1e-2, wd=1e-3 (confirmed with 1st author via email)
    
"""

class CVGZSL(CrossValidation):
    def __init__(self, args):
        super(CVGZSL, self).__init__(args)


    def reproduce_gzsl_improved(self, eeg_1subj, sc_template, block_test, eeg_subjs_pretrain, epochs=[5, 20, 25]):
        ## iGZSL reproduce

        ## Set batch size according to paper
        self.args.batch_size = self.args.seen_num
        lr = 1e-2
        ratio_seen, alpha, beta = self.get_igzsl_params()

        ## Prepare data for training
        trainallblocks, testblocks, trainblocks, validblocks, block_valid = self.split_data_and_extract_seen_gzsl(eeg_1subj, block_test)
        dataloader_train_s1, _, _ = self.convert_data2loader_gzsl(trainblocks, sc_template, is_slide=True) ## finetuning
        # dataloader_trainall_s0, norm_mean, norm_std = self.convert_data2iter_gzsl(trainallblocks, sc_template, is_slide=False) ## fixed window
        # dataloader_train_s0, _, _ = self.convert_data2iter_gzsl(trainblocks, sc_template, is_slide=False) ## cnet
        dataloader_trainall_s0, norm_mean, norm_std = self.convert_data2loader_gzsl(trainallblocks, sc_template, is_slide=False) ## fixed window
        dataloader_train_s0, _, _ = self.convert_data2loader_gzsl(trainblocks, sc_template, is_slide=False) ## cnet

        if self.args.is_pretrain == 1:
            pretrain_blocks = self.extract_seen_pretrain_gzsl(eeg_subjs_pretrain)
            dataloader_pretrain_s1, _, _ = self.convert_data2loader_gzsl(pretrain_blocks, sc_template, is_slide=True)
        else:
            print("***Warning: No pretrain! *** ")

        ## Initialize networks, loss, optimizer
        electrode_num, tp_num = trainblocks['data'].shape[-2], trainblocks['data'].shape[-1]
        net_extraction = ExtractionNet(electrode_num=electrode_num).to(self.args.device)
        net_electrodecomb = ElectrodeCombNet(electrode_num=electrode_num).to(self.args.device)
        group_dataleakage = 4 if self.args.unseen_num > 20 else 8 ## determined by acc of benchmark (stated in GZSL) == leakage
        net_generation = GenerationNet(in_ch=sc_template.shape[-2], group_num=group_dataleakage, 
                                       frequencies=torch.Tensor(self.args.frequencies).to(self.args.device)).to(self.args.device)
        net_transformer = TransformerNet(electrode_num=electrode_num, timepoint_num=tp_num).to(self.args.device)
        net_classification = ClassificationNet(timepoint_num=tp_num, seen_num=self.args.seen_num).to(self.args.device)
        print(f"Model size : TransformerNet={get_trainable_parameter_num(net_transformer)}, " + \
                f"ExtractionNet={get_trainable_parameter_num(net_extraction)}, " + \
                f"GenerationNet(group={group_dataleakage})={get_trainable_parameter_num(net_generation)}, " + \
                f"EleCombNet={get_trainable_parameter_num(net_electrodecomb)}, " + \
                f"ClassificationNet={get_trainable_parameter_num(net_classification)}, ")
        
        # criterion = self.cosine_embedding_loss
        criterion = PearsonCorrelationLoss()
        
        opt = torch.optim.AdamW([{'params': net_electrodecomb.parameters(), 'lr':lr},
                                 {'params': net_generation.parameters(), 'lr': lr},
                                 {'params': net_extraction.parameters(), 'lr': lr},
                                 {'params': net_transformer.parameters(), 'lr': lr},
                                 ], weight_decay=1e-3)
        
        ## Stage 1: Pretrain
        if self.args.is_pretrain == 1:
            epoch_num = 1
            loss_epoch = self.train_4net_igzsl(
                epoch_num, dataloader_pretrain_s1, net_extraction, net_electrodecomb, net_generation, net_transformer, criterion, opt, stage='pretrain')

        ## Stage 2: Fine tuning
        opt = torch.optim.AdamW([{'params': net_electrodecomb.parameters(), 'lr':lr},
                                 {'params': net_generation.parameters(), 'lr': lr},
                                 {'params': net_extraction.parameters(), 'lr': lr},
                                 {'params': net_transformer.parameters(), 'lr': lr},
                                 ], weight_decay=1e-3)
        epoch_num = epochs[0] # true:5
        loss_epoch = self.train_4net_igzsl(
            epoch_num, dataloader_train_s1, net_extraction, net_electrodecomb, net_generation, net_transformer, criterion, opt, stage='finetuning')

        ## Stage3: DSST search (only ClassificationNet training + find Copt using Acc_seen, not including for SST search, since large SST+ST == longer window in online test/decoding, which means our 0.6s Acc is compared with iGZSL (0.6+0.4)s Acc)
        epoch_num = epochs[1]
        criterion_cnet = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(net_classification.parameters(), lr=lr, weight_decay=1e-3)
        loss_cnet = self.train_cnet_gzsl(epoch_num, dataloader_train_s0, net_classification, net_transformer, criterion_cnet, opt)

        # Validate on validblocks
        data_test = np.squeeze(validblocks['data']) # class chnl tp
        data_test = (data_test - norm_mean) / norm_std
        tmpl_sine = np.transpose(sc_template, (0, 2, 1, 3)) # (slide, 2Nh, Nf, Ns)
        template_sine_p0 = torch.from_numpy(tmpl_sine[0,:,:,:].astype('float32')).float().to(self.args.device)
        template_sine_p0 = torch.unsqueeze(template_sine_p0, 0) # (1, 2Nh, Nf, Ns)

        sample_num = data_test.shape[0] # seen
        preds_agg, preds_xfm, preds_ext, preds_cls = [],[],[],[]
        net_extraction.eval()
        net_transformer.eval()
        net_generation.eval()
        net_classification.eval()
        for sample_i in range(sample_num):
            # test_epoch = torch.tensor(data_test[sample_i]).to(self.args.device)
            test_epoch = torch.from_numpy(data_test[sample_i].astype('float32')).float().to(self.args.device)
            test_epoch = torch.unsqueeze(torch.unsqueeze(test_epoch, 0), 0) # (1, 1, Nch, Ns)

            X2 = net_extraction(test_epoch)
            X1, X1_encode = net_transformer(test_epoch)
            S = net_generation(template_sine_p0)
            out_cnet = net_classification(X1_encode)   
            C_agg, C_xfm, C_ext, C_cls = self.predict_4tactics_gzsl(X1, out_cnet, X2, S, beta)
            preds_agg.append(C_agg)
            preds_xfm.append(C_xfm)
            preds_ext.append(C_ext)
            preds_cls.append(C_cls)
        accs = np.array([
            np.mean(preds_agg==np.array(validblocks['label'])),
            np.mean(preds_xfm==np.array(validblocks['label'])),
            np.mean(preds_ext==np.array(validblocks['label'])),
            np.mean(preds_cls==np.array(validblocks['label'])),
        ])
        print(f'Valid: Subject={self.args.subject}, block_valid={block_valid}, Acc=', end='')
        print(accs)
        c_opt_idx = np.argmax(accs)

        ## Fixed training
        # net_xx all trained
        # criterion = self.cosine_embedding_loss
        criterion = PearsonCorrelationLoss()
        opt = torch.optim.AdamW([{'params': net_electrodecomb.parameters(), 'lr':lr},
                                 {'params': net_generation.parameters(), 'lr': lr},
                                 {'params': net_extraction.parameters(), 'lr': lr},
                                 {'params': net_transformer.parameters(), 'lr': lr},
                                 ], weight_decay=1e-3)
        criterion_cnet = nn.CrossEntropyLoss()
        opt_cnet = torch.optim.AdamW(net_classification.parameters(), lr=lr, weight_decay=1e-3)
        epoch_num = epochs[2]
        print('Stage: Fixed training')
        for epoch in range(epoch_num):
            if epoch % 5 < 3:
                loss_cosine = self.train_4net_igzsl(1, dataloader_trainall_s0, net_extraction, net_electrodecomb, net_generation, net_transformer, criterion, opt, is_next_line=False, stage='Fixed window')
            else:
                loss_cnet = self.train_cnet_gzsl(1, dataloader_trainall_s0, net_classification, net_transformer, criterion_cnet, opt_cnet, is_next_line=False)
        
        ## Decode
        data_test = np.squeeze(testblocks['data']) # class chnl tp
        data_test = (data_test - norm_mean) / norm_std
        tmpl_sine = np.transpose(sc_template, (0, 2, 1, 3)) # (slide, 2Nh, Nf, Ns)
        template_sine_p0 = torch.from_numpy(tmpl_sine[0,:,:,:].astype('float32')).float().to(self.args.device)
        template_sine_p0 = torch.unsqueeze(template_sine_p0, 0) # (1, 2Nh, Nf, Ns)

        start = time.time()
        sample_num = data_test.shape[0]
        preds = []
        # preds_agg, preds_xfm, preds_ext, preds_cls = [],[],[],[]
        net_extraction.eval()
        net_transformer.eval()
        net_generation.eval()
        net_classification.eval()
        for sample_i in range(sample_num):
            # test_epoch = torch.tensor(data_test[sample_i]).to(self.args.device)
            test_epoch = torch.from_numpy(data_test[sample_i].astype('float32')).float().to(self.args.device)
            test_epoch = torch.unsqueeze(torch.unsqueeze(test_epoch, 0), 0) # (1, 1, Nch, Ns)

            X2 = net_extraction(test_epoch)
            X1, X1_encode = net_transformer(test_epoch)
            S = net_generation(template_sine_p0)
            out_cnet = net_classification(X1_encode)  

            pred = self.predict_final_igzsl(X1, out_cnet, X2, S, beta, c_opt_idx, is_cond=True)

            preds.append(pred)

        end = time.time()
        print('Computation time=', (end-start)/sample_num)
        acc = np.mean(preds==np.array(testblocks['label']))
        print(f'Subject={self.args.subject}, block_test={block_test}, Acc={acc}')

        metrics = {'acc': acc, 'predicts': preds,
                   'loss_train': 0,
                   'loss_valid': 0, 'loss_unseen': 0, 
                   'corr_train': 0, 'corr_valid': 0, 'corr_unseen': 0}
        return metrics

    

    def predict_final_igzsl(self, X1, out_cnet, X2, S, beta, c_opt_idx, is_cond=True):
        pred_4tactics = self.predict_4tactics_gzsl(X1, out_cnet, X2, S, beta)
        c_agg, c_opt = pred_4tactics[0], pred_4tactics[c_opt_idx]
        if is_cond:
            if c_agg in self.args.label_unseen:
                pred = c_agg
            else:
                pred = c_opt
        else:
            pred = c_agg
        return pred

    def predict_rho_gzsl(self, X1, X2, S, beta):
        rhos1, rhos2 = torch.zeros(self.args.class_num), torch.zeros(self.args.class_num)
        x1, x2 = X1[0,0,0,:].detach(), X2[0,0,0,:].detach()
        for sample_ii in range(self.args.class_num):
            sii = S[0,0,sample_ii,:].detach()
            rhos1[sample_ii] = torch.corrcoef(torch.stack((x1, sii), dim=0))[0, 1]
            rhos2[sample_ii] = torch.corrcoef(torch.stack((x2, sii), dim=0))[0, 1]
        _, pred_xfm = torch.max(rhos1, 0) # -1 will be better
        _, pred_ext = torch.max(rhos2, 0)
        _, pred_agg = torch.max(beta*rhos1+(1-beta)*rhos2, 0)
        # from [0, Nseen-1] to true label in [0, 39]
        # return label_data[pred_agg], label_data[pred_xfm], label_data[pred_ext], c_cls
        return pred_agg.cpu().numpy(), pred_xfm.cpu().numpy(), pred_ext.cpu().numpy()
    
    def predict_cls_gzsl(self, out_cnet):
        ## Ccls
        _, pred_cls = torch.max(out_cnet, 1)
        c_cls = self.args.label_seen[pred_cls]
        return c_cls
    
    def predict_4tactics_gzsl(self, X1, out_cnet, X2, S, beta):
        agg, xfm, ext = self.predict_rho_gzsl(X1, X2, S, beta)
        c_cls = self.predict_cls_gzsl(out_cnet)
        return agg, xfm, ext, c_cls
        

    def train_cnet_gzsl(self, epoch_num, dataloader, net_classification, net_transformer, criterion_cnet, opt, is_next_line=True):
        print("Training data size:", dataloader.dataset.x_mean.shape, 'bsz=', dataloader.batch_size)
        # Train
        for epoch in range(epoch_num):
            net_classification.train()
            net_transformer.eval()
            loss_epoch = 0 # loss_epoch1, loss_epoch2 = 0, 0
            t1 = time.time()
            for batch_ii, batch_data in enumerate(dataloader):
                x_batch, mean_batch, sine_batch, label_batch = batch_data
                x_batch, mean_batch, sine_batch, label_batch = x_batch.to(self.args.device), mean_batch.to(self.args.device), sine_batch.to(self.args.device), label_batch.to(self.args.device)
                X1, X1_encode = net_transformer(x_batch) # (batch, 1, 1, Ns)
                X1_encode = X1_encode.detach()
                out = net_classification(X1_encode)
                loss = criterion_cnet(out, label_batch) # now label = [0, seen-1]; prev: label not from 0-seen-1
                _, pred = torch.max(out, 1)
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_epoch += loss
            delta_t = time.time() - t1
            print(f"\rCNet Epoch {epoch}: Loss: {loss_epoch:4f} (Time={delta_t:.2f})", end='')
        if is_next_line:
            print()
        # return net_classification
        return loss_epoch

    def train_4net_igzsl(self, epoch_num, dataloader, net_extraction, net_electrodecomb, net_generation, net_transformer, criterion, opt, is_next_line=True, stage='finetuning'):
        ## label_batch [0, Nseen-1], not true label! so, S1 should be self.label_seen[sii]
        if is_next_line: # whole training, not 1 step
            # print("Training data size:", dataloader.dataset.tensors[0].shape, 'bsz=', dataloader.batch_size)
            print("Training data size:", dataloader.dataset.x_mean.shape, 'bsz=', dataloader.batch_size)
            
        for epoch in range(epoch_num):
            net_extraction.train()
            net_electrodecomb.train()
            net_generation.train()
            net_transformer.train()
            loss_epoch = 0 # loss_epoch1, loss_epoch2 = 0, 0
            t1 = time.time()
            for batch_ii, batch_data in enumerate(dataloader):
                x_batch, mean_batch, sine_batch, label_batch = batch_data
                x_batch, mean_batch, sine_batch, label_batch = x_batch.to(self.args.device), mean_batch.to(self.args.device), sine_batch.to(self.args.device), label_batch.to(self.args.device)
                batch_num = x_batch.shape[0]
                tmp = torch.ones(batch_num).to(self.args.device)
                X2 = net_extraction(x_batch) # (batch, 1, 1, Ns)
                X1, X1_encode = net_transformer(x_batch) # (batch, 1, 1, Ns)
                T = net_electrodecomb(mean_batch) # (batch, 1, 1, Ns)
                S = net_generation(sine_batch) # (batch, 1, Nf, Ns)

                # S1 = [S[sii,:,self.args.label_batch[sii],:] for sii in range(S.shape[0])] # bug true label==label_seen[label_batch[bii]]
                S1 = [S[bii,:,self.args.label_seen[label_batch[bii]],:] for bii in range(batch_num)]
                S1 = torch.stack(S1, dim=0)
                S1 = torch.unsqueeze(S1, dim=1)
                loss3 = criterion(S1, T)
                loss1 = criterion(X1, T)
                loss2 = criterion(X2, T)
                loss = loss1 + loss2 + loss3
                opt.zero_grad()
                loss.backward()
                opt.step()

                loss_epoch += loss

            delta_t = time.time() - t1
            print(f"\r{stage}: Epoch {epoch}: Loss: {loss_epoch:4f} (Time={delta_t:.2f})", end='')
        if is_next_line:
            print()
        return loss_epoch

    
    # def reproduce_gzsl_fb(self, eeg_1subj, sc_template, block_test):
    #     filterbank = Filterbank(filterbank_num=self.args.fb_num, sampling_rate=self.args.sampling_rate)
    #     rhos_alltest = []
    #     losses, corrs = [], []
    #     for fbii in range(self.args.fb_num):
    #         self.args.fbii = fbii
    #         eeg_fb = filterbank.filter_subband(eeg_1subj, fbii)
    #         # eeg_subjs_pretrain_fb = filterbank.filter_subband(eeg_subjs_pretrain, fbii) if self.args.is_pretrain else []

    #         # metrics_fbii, rhos_fbii = self.leave_one_block_out(eeg_fb, sc_template, block_test, eeg_subjs_pretrain_fb)
    #         metrics_fbii, rhos_fbii = self.reproduce_gzsl(eeg_fb, sc_template, block_test)
    #         rhos_alltest.append(rhos_fbii)

    #         losses.append([metrics_fbii['loss_train'], metrics_fbii['loss_valid'], metrics_fbii['loss_unseen']])
    #         corrs.append([metrics_fbii['corr_train'], metrics_fbii['corr_valid'], metrics_fbii['corr_unseen']])

    #     ## Combine FB
    #     rhos_alltest = np.stack(rhos_alltest, axis=-1) # test class fb
    #     rho_vec = np.matmul(rhos_alltest, filterbank.fb_weights) # test class
    #     predicts = np.argmax(rho_vec, axis=1)
    #     acc = np.mean(predicts==np.array(self.args.labels))
    #     print(f'Subject={self.args.subject}, block_test={block_test}, Acc={acc:.4f}')

    #     losses_mean = np.mean(np.array(losses), axis=0)
    #     corrs_mean = np.mean(np.array(corrs), axis=0)
    #     metrics = {'loss_train': losses_mean[0], 'loss_valid': losses_mean[1], 'loss_unseen': losses_mean[2],
    #                'corr_train': corrs_mean[0], 'corr_valid': corrs_mean[1], 'corr_unseen': corrs_mean[2],
    #                'acc': acc}
    #     return metrics

    def reproduce_gzsl(self, eeg_1subj, sc_template, block_test):
        # blks: train & test
        assert self.args.batch_size == 32
        trainblocks, testblocks = self.separate_all_into_train_test(eeg_1subj, block_test, self.args.blocks)

        # train --> seen
        data = trainblocks['data'][:,:,self.args.label_seen,:,:] # (blk,slide,Nf_seen,Nch, Ns)
        labels = self.args.label_seen # (Nf_seen,)

        ## Normalize
        norm_mean, norm_std = np.mean(data), np.std(data)
        data = (data - norm_mean) / norm_std

        # templates
        tmpl_mean = np.mean(data, axis=0) # (slide,Nf_seen,Nch,Ns)
        tmpl_sine = sc_template # (slide,Nf,2Nh,Ns)
        tmpl_sine = np.transpose(tmpl_sine, (0, 2, 1, 3)) # (slide, 2Nh, Nf, Ns)

        # repeat to align
        labels = [labels for _ in range(data.shape[0]*data.shape[1])]
        labels = np.stack(labels, axis=0) # (blk*slide, Nf_seen)
        template_mean = [tmpl_mean for _ in range(data.shape[0])]
        template_mean = np.stack(template_mean, axis=0) # (blk, slide, Nf_seen, Nch, Ns)
        template_sine = [tmpl_sine for _ in range(data.shape[0])]
        template_sine = np.stack(template_sine, axis=0)
        template_sine = [template_sine for _ in range(data.shape[2])]
        template_sine = np.stack(template_sine, axis=2) # (blk, slide, Nf_seen, 2Nh, Nf, Ns)

        # reshape (samples = blks*slide*Nf_seen)
        data = np.reshape(data, (-1, data.shape[-2], data.shape[-1])) 
        data = np.expand_dims(data, axis=-3) # (samples, 1, Nch, Ns)
        labels = np.reshape(labels, (-1))
        template_mean = np.reshape(template_mean, (-1, data.shape[-2], data.shape[-1]))
        template_mean = np.expand_dims(template_mean, axis=-3) # (samples, 1, Nch, Ns)
        template_sine = np.reshape(template_sine, (-1, template_sine.shape[-3], template_sine.shape[-2], template_sine.shape[-1])) # (samples, 2Nh, Nf, Ns)

        # prepare for training
        X_train = torch.from_numpy(data.astype('float32')).float().to(self.args.device)
        X_mean = torch.from_numpy(template_mean.astype('float32')).float().to(self.args.device)
        X_sine = torch.from_numpy(template_sine.astype('float32')).float().to(self.args.device)
        labels = torch.tensor(labels).type(torch.LongTensor)

        dataset = TensorDataset(X_train, X_mean, X_sine, labels)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

        electrode_num = X_train.shape[-2]
        frequencies = torch.Tensor(self.args.frequencies).to(self.args.device)
        net_extraction = ExtractionNet(electrode_num=electrode_num).to(self.args.device)
        net_electrodecomb = ElectrodeCombNet(electrode_num=electrode_num).to(self.args.device)
        net_generation = GenerationNet(in_ch=X_sine.shape[-3], group_num=8, frequencies=frequencies).to(self.args.device)
        
        criterion1 = self.cosine_embedding_loss
        criterion2 = self.cosine_embedding_loss
        criterion3 = self.cosine_embedding_loss

        # opt
        opt1 = torch.optim.AdamW(net_extraction.parameters(), lr=1e-2, weight_decay=1e-3)
        opt2 = torch.optim.AdamW([{'params': net_electrodecomb.parameters(), 'lr':1e-2},
                                {'params': net_generation.parameters(), 'lr': 1e-2}], weight_decay=1e-3)
        scheduler1 = torch.optim.lr_scheduler.MultiStepLR(opt1, milestones=[14], gamma=0.1)
        scheduler2 = torch.optim.lr_scheduler.MultiStepLR(opt2, milestones=[14], gamma=0.1)

        epoch_num1 = 20

        for epoch in range(epoch_num1):
            net_extraction.train()
            net_electrodecomb.train()
            net_generation.train()
            loss_epoch1, loss_epoch2 = 0, 0
            t1 = time.time()
            for batch_ii, batch_data in enumerate(dataloader):
                at_batch = time.time()
                x_batch, mean_batch, sine_batch, label_batch = batch_data
                batch_num = x_batch.shape[0]
                tmp = torch.ones(batch_num).to(self.args.device)
                X = net_extraction(x_batch)
                T = net_electrodecomb(mean_batch) # (batch, 1, 1, Ns)
                S = net_generation(sine_batch) # (batch, 1, Nf, Ns)
                    
                loss1 = criterion1(X, T.detach())
                # loss1 = criterion1(torch.squeeze(X), torch.squeeze(T.detach()), tmp)
                opt1.zero_grad()
                loss1.backward()
                opt1.step()
                loss_epoch1 += loss1

                # idx0 = torch.arange(0, label_batch.shape[0]).unsqueeze(1)
                # idx2 = label_batch.unsquueze(1)
                # S1 = S[idx0, :, idx2, :]
                S1 = [S[sii,:,label_batch[sii],:] for sii in range(S.shape[0])]
                S1 = torch.stack(S1, dim=0)
                S1 = torch.unsqueeze(S1, dim=1)
                loss2 = criterion2(T, S1)
                # loss2 = criterion2(torch.squeeze(T), torch.squeeze(S1), tmp)
                opt2.zero_grad()
                loss2.backward()
                opt2.step()

                loss_epoch2 += loss2
                at_batch = time.time() - at_batch
            delta_t = time.time() - t1
            print(f"\rStage1 Epoch {epoch}: Loss1: {loss_epoch1:4f} (lr={scheduler1.get_last_lr()}), Loss2: {loss_epoch2:.4f} (lr={scheduler2.get_last_lr()}) (Time={delta_t:.2f})", end='')
            scheduler1.step()
            scheduler2.step()
        print()

        # Stage2
        epoch_num2 = 10
        # opt3 = torch.optim.AdamW([{'params': net_extraction.parameters(), 'lr':1e-4},
        #                         {'params': net_generation.parameters(), 'lr': 1e-4}], weight_decay=1e-3)
        lr = 1e-4
        opt3 = torch.optim.AdamW(net_generation.parameters(), lr=lr, weight_decay=1e-3)
        for epoch in range(epoch_num2):
            net_extraction.train()
            # net_electrodecomb.train()
            net_generation.train()
            loss_epoch3 = 0
            t1 = time.time()
            for batch_ii, batch_data in enumerate(dataloader):
                x_batch, mean_batch, sine_batch, label_batch = batch_data
                batch_num = x_batch.shape[0]
                tmp = torch.ones(batch_num).to(self.args.device)
                X = net_extraction(x_batch) # (ba)
                T = net_electrodecomb(mean_batch) # (batch, 1, 1, Ns)
                S = net_generation(sine_batch) # (batch, 1, Nf, Ns)

                S1 = [S[sii,:,label_batch[sii],:] for sii in range(S.shape[0])]
                S1 = torch.stack(S1, dim=0)
                S1 = torch.unsqueeze(S1, dim=1)
                loss3 = criterion3(X, S1)
                # loss3 = criterion3(torch.squeeze(X), torch.squeeze(S1), tmp)
                opt3.zero_grad()
                loss3.backward()
                opt3.step()
                loss_epoch3 += loss3
            delta_t = time.time() - t1
            print(f"\rStage2 Epoch {epoch}: Loss3: {loss_epoch3:4f} (lr={lr})(Time={delta_t:.2f})", end='')
        print()
        a = 1
        ## Decode
        data_test = np.squeeze(testblocks['data']) # class chnl tp
        data_test = (data_test - norm_mean) / norm_std
        template_sine_p0 = torch.from_numpy(tmpl_sine[0,:,:,:].astype('float32')).float().to(self.args.device)
        template_sine_p0 = torch.unsqueeze(template_sine_p0, 0) # (1, 2Nh, Nf, Ns)

        sample_num = data_test.shape[0]
        # data_test = np.matmul(weights_sf_seen, data_test) ## preprocessing
        class_num = template_sine_p0.shape[-2]
        predicts = []
        net_extraction.eval()
        net_generation.eval()
        rho_mat = np.zeros((class_num, class_num))
        for sample_i in range(sample_num):
            # test_epoch = torch.tensor(data_test[sample_i]).to(self.args.device)
            test_epoch = torch.from_numpy(data_test[sample_i].astype('float32')).float().to(self.args.device)
            test_epoch = torch.unsqueeze(torch.unsqueeze(test_epoch, 0), 0) # (1, 1, Nch, Ns)
            X = net_extraction(test_epoch)
            S = net_generation(template_sine_p0)
            rho_vec = np.zeros(class_num)
            loss_vec = np.zeros(class_num)
            for class_ii in range(class_num):
                x = X[0,0,0,:].detach()
                sii = S[0,0,class_ii,:].detach()
                # loss_test = criterion3(torch.unsqueeze(x, 0), torch.unsqueeze(sii, 0), torch.ones(1).to(self.args.device))
                loss_test = criterion3(x, sii)
                tmp = torch.stack((x, sii), dim=0) # (2, Ns)
                rho = torch.corrcoef(tmp)[0, 1]

                rho_vec[class_ii] = rho.detach().cpu().numpy()
                loss_vec[class_ii] = loss_test
                # rho = np.corrcoef(X[0,0,0,:], S[0,0,class_ii,:])
                # rho_vec[class_ii] = rho[0, 1]
            predict = np.argmax(rho_vec)
            predicts.append(predict)
            rho_mat[sample_i, :] = rho_vec
            
        acc = np.mean(predicts==np.array(testblocks['label']))
        print(f'Subject={self.args.subject}, block_test={block_test}, Acc={acc:.4f}')

        metrics = {'acc': acc, 'predicts': predicts,
                   'loss_train': loss_epoch3.detach().cpu().numpy(),
                   'loss_valid': 0, 'loss_unseen': 0, 
                   'corr_train': 0, 'corr_valid': 0, 'corr_unseen': 0}
        return metrics, rho_mat

    def extract_seen_pretrain_gzsl(self, eeg_pretrain):
        trainblocks = {
            'data': eeg_pretrain,
            'label': self.args.labels,
            'block': self.args.blocks,
        }
        trainblocks = self.extract_seen_data_gzsl(trainblocks)
        return trainblocks

    def split_data_and_extract_seen_gzsl(self, eeg_1subj, block_test):
        """
        Separate dataset
        Inputs:
            eeg_1subj - ndarray (block slide class chnl tp)
            block_test - list: 1 test block for lobo
        Returns:
            trainallblocks - dict: (N-1) blocks
            testblocks - dict: 1 block
            trainblocks: (N-2) blocks extracted from trainallblocks
            validblocks: 1 block extracted from trainallblocks
            block_valid: randomly chosen block for validation
        """
        trainallblocks, testblocks = self.separate_all_into_train_test(eeg_1subj, block_test, self.args.blocks)

        ## Separate blks: train & valid
        block_trainall = trainallblocks['block']
        block_valid = random.choice(block_trainall)
        trainblocks, validblocks = self.separate_all_into_train_test(trainallblocks['data'], [block_valid], block_trainall)

        ## Extract seen class in train & valid
        trainblocks = self.extract_seen_data_gzsl(trainblocks)
        validblocks = self.extract_seen_data_gzsl(validblocks)
        trainallblocks = self.extract_seen_data_gzsl(trainallblocks)
        return trainallblocks, testblocks, trainblocks, validblocks, block_valid


    def extract_seen_data_gzsl(self, datablocks):
        ## Extract data of seen classes only in datablocks, delete unseen data
        if len(datablocks['data'].shape) == 5: # train
            datablocks['data'] = datablocks['data'][:,:,self.args.label_seen,:,:]
        elif len(datablocks['data'].shape) == 4: # valid
            datablocks['data'] = datablocks['data'][:,self.args.label_seen,:,:]
        else:
            raise ValueError("datablocks['data'] shape error!")
        
        datablocks['label'] = self.args.label_seen
        return datablocks

    # def convert_data2iter_gzsl(self, trainblocks, sc_template, is_slide=True):
    #     ## Convert dataset to torch dataloader for training
    #     def largest_power_of_two(a):
    #         b = 1
    #         while a > 1:
    #             a >>= 1
    #             b <<= 1
    #         return b
    #     k = largest_power_of_two(int(1 / (self.args.step*0.01))) if is_slide else 1
    #     k = k * 2 if self.args.is_pretrain == 1 else k
    #     X_train, X_mean, X_sine, labels, norm_mean, norm_std = self.convert_nd22d_gzsl(trainblocks, sc_template, is_slide)
    #     dataset = TensorDataset(X_train, X_mean, X_sine, labels)
    #     dataloader = DataLoader(dataset, batch_size=self.args.batch_size*k, shuffle=True)
    #     return dataloader, norm_mean, norm_std


    def convert_data2loader_gzsl(self, trainblocks, sc_template, is_slide=True):
        """
        Convert dataset to numpy dataloader for training (torch.from_numpy() in training)
        Returns:
            dataloader: iterator for training, data stored in CPU memory
            norm_mean/std: for normalization in testing stage
        """
        ## Multiply batchsize by k if sliding for faster training
        def largest_power_of_two(a):
            b = 1
            while a > 1:
                a >>= 1
                b <<= 1
            return b
        k = largest_power_of_two(int(1 / (self.args.step*0.01))) if is_slide else 1
        k = k * 2 if self.args.is_pretrain == 1 else k

        # dataset, norm_mean, norm_std = self.convert_signal2dataset_gzsl(trainblocks, sc_template, is_slide=True) # bug
        dataset, norm_mean, norm_std = self.convert_signal2dataset_gzsl(trainblocks, sc_template, is_slide=is_slide)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size*k, shuffle=True, num_workers=1)
        return dataloader, norm_mean, norm_std

    def convert_signal2dataset_gzsl(self, trainblocks, sc_template, is_slide=True):
        """
        Format GZSLdataset for dataloader generation.
        data contained in dataloader:
            data: single trial EEG
            template_mean: trial averaged EEG
            template_sine: sine-cosine artificial template
            labels: class label for each `data`
        """
        data = trainblocks['data']
        labels = np.arange(0, len(trainblocks['label'])) # [0, seen-1]; raw label into cross-entropy raise Error

        ## Normalize
        norm_mean, norm_std = np.mean(data), np.std(data)
        data = (data - norm_mean) / norm_std # (blk, slide, Nf_seen, Nch, Ns)

        # templates
        tmpl_mean = np.mean(data, axis=0) # (slide,Nf_seen,Nch,Ns)
        tmpl_sine = sc_template # (slide,Nf,2Nh,Ns)
        tmpl_sine = np.transpose(tmpl_sine, (0, 2, 1, 3)) # (slide, 2Nh, Nf, Ns)

        # repeat to align
        labels = [labels for _ in range(data.shape[1])]
        labels = np.stack(labels, axis=0) # (slide, Nf_seen)
        labels = [labels for _ in range(data.shape[0])]
        labels = np.stack(labels, axis=0) # (blk, slide, Nf_seen)

        template_mean = [tmpl_mean for _ in range(data.shape[0])]
        template_mean = np.stack(template_mean, axis=0) # (blk, slide, Nf_seen, Nch, Ns)
        
        template_sine = [tmpl_sine for _ in range(data.shape[0])]
        template_sine = np.stack(template_sine, axis=0)
        template_sine = [template_sine for _ in range(data.shape[2])]
        template_sine = np.stack(template_sine, axis=2) # (blk, slide, Nf_seen, 2Nh, Nf, Ns)

        if not is_slide:
            data = data[:,0,:,:,:]
            labels = labels[:,0,:]
            template_mean = template_mean[:,0,:,:,:]
            template_sine = template_sine[:,0,:,:,:,:]

        # reshape (samples = blks*slide*Nf_seen)
        data = np.reshape(data, (-1, data.shape[-2], data.shape[-1])) 
        data = np.expand_dims(data, axis=-3) # (samples, 1, Nch, Ns)
        labels = np.reshape(labels, (-1))
        template_mean = np.reshape(template_mean, (-1, data.shape[-2], data.shape[-1]))
        template_mean = np.expand_dims(template_mean, axis=-3) # (samples, 1, Nch, Ns)
        template_sine = np.reshape(template_sine, (-1, template_sine.shape[-3], template_sine.shape[-2], template_sine.shape[-1])) # (samples, 2Nh, Nf, Ns)

        dataset = GZSLDataset(data, template_mean, template_sine, labels)
        return dataset, norm_mean, norm_std

    # def convert_nd22d_gzsl(self, trainblocks, sc_template, is_slide=True):
    #     data = trainblocks['data']
    #     labels = np.arange(0, len(trainblocks['label'])) # [0, seen-1]; raw label into cross-entropy raise Error

    #     ## Normalize
    #     norm_mean, norm_std = np.mean(data), np.std(data)
    #     data = (data - norm_mean) / norm_std # (blk, slide, Nf_seen, Nch, Ns)

    #     # templates
    #     tmpl_mean = np.mean(data, axis=0) # (slide,Nf_seen,Nch,Ns)
    #     tmpl_sine = sc_template # (slide,Nf,2Nh,Ns)
    #     tmpl_sine = np.transpose(tmpl_sine, (0, 2, 1, 3)) # (slide, 2Nh, Nf, Ns)

    #     # repeat to align
    #     labels = [labels for _ in range(data.shape[1])]
    #     labels = np.stack(labels, axis=0) # (slide, Nf_seen)
    #     labels = [labels for _ in range(data.shape[0])]
    #     labels = np.stack(labels, axis=0) # (blk, slide, Nf_seen)
    #     template_mean = [tmpl_mean for _ in range(data.shape[0])]
    #     template_mean = np.stack(template_mean, axis=0) # (blk, slide, Nf_seen, Nch, Ns)
    #     template_sine = [tmpl_sine for _ in range(data.shape[0])]
    #     template_sine = np.stack(template_sine, axis=0)
    #     template_sine = [template_sine for _ in range(data.shape[2])]
    #     template_sine = np.stack(template_sine, axis=2) # (blk, slide, Nf_seen, 2Nh, Nf, Ns)

    #     if not is_slide:
    #         data = data[:,0,:,:,:]
    #         labels = labels[:,0,:]
    #         template_mean = template_mean[:,0,:,:,:]
    #         template_sine = template_sine[:,0,:,:,:,:]

    #     # reshape (samples = blks*slide*Nf_seen)
    #     data = np.reshape(data, (-1, data.shape[-2], data.shape[-1])) 
    #     data = np.expand_dims(data, axis=-3) # (samples, 1, Nch, Ns)
    #     labels = np.reshape(labels, (-1))
    #     template_mean = np.reshape(template_mean, (-1, data.shape[-2], data.shape[-1]))
    #     template_mean = np.expand_dims(template_mean, axis=-3) # (samples, 1, Nch, Ns)
    #     template_sine = np.reshape(template_sine, (-1, template_sine.shape[-3], template_sine.shape[-2], template_sine.shape[-1])) # (samples, 2Nh, Nf, Ns)

    #     # prepare for training
    #     X_train = torch.from_numpy(data.astype('float32')).float().to(self.args.device)
    #     X_mean = torch.from_numpy(template_mean.astype('float32')).float().to(self.args.device)
    #     X_sine = torch.from_numpy(template_sine.astype('float32')).float().to(self.args.device)
    #     labels = torch.tensor(labels).type(torch.LongTensor).to(self.args.device)

    #     return X_train, X_mean, X_sine, labels, norm_mean, norm_std
    
    def get_igzsl_params(self):
        ratio_seen = self.args.seen_num / self.args.class_num
        alpha = self.get_igzsl_alpha(ratio_seen)
        beta = self.get_igzsl_beta(ratio_seen)
        return ratio_seen, alpha, beta
    
    def get_igzsl_alpha(self, ratio):
        if ratio <= 0.2:
            alpha = 0.6
        elif ratio < 0.8:
            alpha = 1/2*ratio + 1/2
        else:
            alpha = 0.9
        return alpha

    def get_igzsl_beta(self, ratio):
        if ratio <= 0.2:
            beta = 0
        elif ratio < 0.8:
            beta = 5/6*ratio - 1/6
        else:
            beta = 0.5
        return beta

    # def test_util(self):
    #     ## Test
    #     iters = int(1e6)
    #     device = 'cuda'
    #     x = torch.randn((64, 1, 9, 250)).to(device)
    #     sine = torch.randn((64, 10, 40, 250)).to(device)

    #     net1 = ExtractionNet(9).to(device) # 0.28s
    #     net2 = TemporalConvNet(1).to(device) # 4.14
    #     net3 = TemporalConvNet(1, num_channels=[16]*1).to(device) # 0.4466s 
    #     net4 = GenerationNet(in_ch=10, group_num=8, frequencies=torch.tensor(np.linspace(8, 15.8, 40)).to(device)).to(device)


    #     t1 = time.time()
    #     for _ in range(iters):
    #         # out = net1(x)
    #         # out = net2(x)
    #         # out = net3(x)
    #         out = net4(sine)

    #     t2 = time.time()
    #     print(t2-t1)
    #     # End test

class GZSLDataset(Dataset):
    def __init__(self, x_train, x_mean, x_sine, labels):
        self.x_train = x_train
        self.x_mean = x_mean
        self.x_sine = x_sine
        self.labels = labels

    def __len__(self):
        return len(self.x_train)
    
    def __getitem__(self, index):
        s_train = torch.tensor(self.x_train[index], dtype=torch.float32)
        s_mean = torch.tensor(self.x_mean[index], dtype=torch.float32)
        s_sine = torch.tensor(self.x_sine[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.long)

        return s_train, s_mean, s_sine, label
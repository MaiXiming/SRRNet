import os
import numpy as np
import argparse
import torch
from datetime import datetime
 
is_jupyter = False
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from PrepareData.SSVEPLoader import *
from utils import *
from CV.CrossValidation import *
from CV.CVGZSL import *
from CV.CVRegression import *


def main():
    args = set_args() ## cmd args
    random_fix(args.random_seed, is_cudnn_fixed=(args.is_cudnn_fixed==1))
    ## Load eeg data
    eeg_subj, sc_template, eeg_subjs_pretrain, args = load_ssvep_data(args)
    
    ## Setup CV
    cv = set_crossvalidation(args)

    ## Recordings save
    csv_path, pkl_path = name_record_files(args)
    if args.subject == 0 and (args.is_csv_output==1): # first fold
        print_args(args)
        init_recordings(csv_path, pkl_path, args)

    ## Create .pickle file to save the details new
    # a = 1
    # data = {
    #     'args': args,
    #     'recon_templates': np.zeros((args.unseen_num, args.fb_num, args.chnl_num, args.timepoint_num, )),
    #     'true_templates': np.zeros((args.unseen_num, args.fb_num, args.chnl_num, args.timepoint_num, )),
    #     'test_results': np.zeros((args.block_num*args.class_num, 2)), # (true, predict) * blk*cls,
    #     'confusemat': np.zeros((args.class_num, args.class_num)),
    #     'acc': 0.0,
    # }
    # path_suffix = './Outputs/Results/Details'
    # ensure_path(path_suffix)
    # fn_detail = f"{args.dataset}-u{args.unseen_num}-t{args.window:.1f}-s{args.subject}.pickle"
    # fn_detail = os.path.join(path_suffix, fn_detail)
    # file = open(fn_detail, 'wb')
    # pickle.dump(data, file)
    # file.close()
    # args.fn_detail = fn_detail
    if args.model == 'srrnet':
        args = create_file_detail(args)

    


    ## Leave-one-block-out CV: Train & Test
    accs, losses, corrs = [], [], []
    confusemat = np.zeros((args.class_num, args.class_num)) # y-true; x-predict; sum(each row)==block_num
    for testii in args.blocks:
        block_test = [testii]
        if args.model in ['trca', 'tdca']:
            metrics = cv.lobo_tradition(eeg_subj, block_test)

        elif args.model[-4:] == 'gzsl':
            if args.model == 'igzsl':
                metrics = cv.reproduce_gzsl_improved(eeg_subj, sc_template, block_test, eeg_subjs_pretrain)
            elif args.model == 'gzsl' and args.fb_num != 1:
                metrics = cv.reproduce_gzsl_fb(eeg_subj, sc_template, block_test)
            elif args.model == 'gzsl':
                metrics, _ = cv.reproduce_gzsl(eeg_subj, sc_template, block_test)
            else:
                raise ValueError("GZSL condition error")
        else: # our framework
            if args.fb_num == 1:
                metrics, _ = cv.lobo_regress(eeg_subj, sc_template, block_test, eeg_subjs_pretrain)
            else: # FB
                metrics = cv.lobo_fb(eeg_subj, sc_template, block_test, eeg_subjs_pretrain)
    
        accs.append(metrics['acc'])
        losses.append([metrics['loss_train'], metrics['loss_valid'], metrics['loss_unseen']])
        corrs.append([metrics['corr_train'], metrics['corr_valid'], metrics['corr_unseen']])
        for tii, pii in enumerate(metrics['predicts']):
            confusemat[tii, pii] += 1
        
    ## Record results
    acc_mean = np.mean(accs)
    losses_mean = np.mean(np.array(losses), axis=0)
    corrs_mean = np.mean(np.array(corrs), axis=0)
    print(f'Subject={args.subject},  Mean Acc={acc_mean:.4f}, Mean unseen corr: {corrs_mean[2]:.4f}') # 
    if args.is_csv_output==1:
        update_csv(csv_path, pkl_path, acc_mean, losses_mean, corrs_mean, confusemat, args)


    ## Records details
    if args.model == 'srrnet':
        with open(args.fn_detail, 'rb') as file:
            data = pickle.load(file)
        data['acc'] = acc_mean
        data['confusemat'] = confusemat

        file = open(args.fn_detail, 'wb')
        pickle.dump(data, file)
        file.close()


def set_args():
    """Create cmd arguments for command line execution"""
    parser = argparse.ArgumentParser()
    ## Settings
    parser.add_argument('--dataset', type=str, default='benchmark') # benchmark beta
    parser.add_argument('--subjects', type=int, default=35) 
    parser.add_argument('--unseen-num', type=int, default=8)
    parser.add_argument('--window', type=float, default=1)
    parser.add_argument('--subject', type=int, default=31)
    parser.add_argument('--spatialfilter', type=str, default='trca') # trca; tdca
    parser.add_argument('--model', type=str, default='srrnet')
    # srrnet, gzsl, igzsl, trca, tdca, srrv2
    
    parser.add_argument('--step', type=float, default=25) # (0,1): second; [1,100]: win%
    parser.add_argument('--trad-trainblocks', type=int, default=5)
    parser.add_argument('--is-pretrain', type=int, default=0)
    parser.add_argument('--is-plot-template', type=int, default=1) # 1 in debug, 0 in nfolds
    parser.add_argument('--is-losscurve', type=int, default=0) # 1 in debug, 0 in nfolds
    parser.add_argument('--is-plot-weights', type=int, default=0) # 1 in debug, 0 in nfolds
    parser.add_argument('--is-csv-output', type=int, default=0) # 1 in debug, 0 in nfolds
    parser.add_argument('--is-cudnn-fixed', type=int, default=1) # gzsl == 0
    parser.add_argument('--is-tsne-data', type=int, default=0) # gzsl == 0

    ## Hyperparams
    parser.add_argument('--fb-num', type=int, default=5) # 1==noFB, 5==FB
    parser.add_argument('--harmonic-num', type=int, default=5) # gzsl:3 else:5
    parser.add_argument('--lr', type=float, default=1e-3) # 1e-3
    parser.add_argument('--batch-size', type=int, default=64) 
    parser.add_argument('--epoch-num', type=int, default=200) 
    
    parser.add_argument('--lr-pretrain', type=float, default=1e-3)
    parser.add_argument('--batch-size-pretrain', type=int, default=64)
    parser.add_argument('--epoch-num-pretrain', type=int, default=50) # 10
    
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--loss', type=str, default='mse') # mse > corr
    parser.add_argument('--is-earlystopping', type=int, default=1) ## default: 0
    parser.add_argument('--es-patience', type=int, default=100)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.1)
    
    ## Params (fixed usually)
    parser.add_argument('--is-filter', type=int, default=1) 
    parser.add_argument('--is-detrend', type=int, default=0) 
    parser.add_argument('--is-normalize', type=int, default=1)
    parser.add_argument('--is-phase-harmonic', type=int, default=1)
    parser.add_argument('--is-tmpl-trueseen', type=int, default=1) # seen tmpl: 1==true; 0==regressed
    parser.add_argument('--save-path', type=str, default='./Outputs/checkpoints')
    parser.add_argument('--timenow', type=str, default='-1') # set in nfolds / below
    parser.add_argument('--server', type=str, default='default', help='default or pc')
    parser.add_argument('--subject-condition', type=str, default='intra')
    parser.add_argument('--random-seed', type=int, default=202403) 

    parser.add_argument('--nfolds', type=int, default=0) # python onefold.py in debug (0) or in ngpu (1)
    
    if is_jupyter:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.timenow == '-1':
        timenow = datetime.now()
        args.timenow = timenow.strftime("%Y%m%d-%H%M%S")
    
    ## Path
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    args.save_path = os.path.join(args.save_path, f'{args.subject_condition}_subj{args.subject}_{args.timenow}')
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    args.save_path = os.path.join(args.save_path, 'best-mdl.pth')
    args.load_path = os.path.join(args.save_path, 'best-mdl')

    ensure_path('Tmp/')
    ensure_path('Tmp/logs')
    ensure_path('Tmp/fig_loss')
    ensure_path('Tmp/fig_tmpl')

    ## subjects:
    subjects_pretrain = []
    for sii in range(args.subjects):
        if sii != args.subject:
            subjects_pretrain.append(sii)
    args.subjects_pretrain = subjects_pretrain

    return args


def set_unseen_valid_labels(args):
    args.labels = list(range(args.class_num))
    args.blocks = list(range(args.block_num))

    if args.unseen_num == 8:
        label_unseen, label_valid = list(range(16, 24)), [8,9,10,11,28,29,30,31] # 8 unseen
    elif args.unseen_num == 20:
        label_unseen, label_valid = list(range(10, 30)), [0,4,9,30,39] # 20 unseen
    elif args.unseen_num == 32:
        label_unseen, label_valid = list(range(4, 36)), [1, 38] # 32 unseen
    elif args.unseen_num == 36:
        label_unseen, label_valid = list(range(2, 38)), [1] # 36 unseen
    # label_unseen, label_valid = list(range(6, 34)), [2, 4, 37] # 28 unseen
    else:
        raise ValueError("args.unseen_num not found!")

    args.label_unseen = label_unseen
    args.label_valid = label_valid

    args.label_seen = [item for item in args.labels if item not in args.label_unseen]
    args.label_train = [item for item in args.label_seen if item not in args.label_valid]
    # return label_unseen, label_valid

    args.seen_num = args.class_num - args.unseen_num

    if args.dataset == 'beta':
        labels_bm2bt = find_label_bm2bt()
        args.label_unseen = [labels_bm2bt[i] for i in args.label_unseen]
        args.label_valid = [labels_bm2bt[i] for i in args.label_valid]
        args.label_seen = [labels_bm2bt[i] for i in args.label_seen]
        args.label_train = [labels_bm2bt[i] for i in args.label_train]
    return args

def set_crossvalidation(args):
    ## Cross validation prepare
    if args.model == 'gzsl' or args.model == 'igzsl':
        cv = CVGZSL(args)
        assert args.harmonic_num == 3
        assert args.is_cudnn_fixed == 0
    elif args.model in ['trca', 'tdca']:
        cv = CrossValidation(args)
    else:
        cv = CVRegression(args)
    return cv

def load_ssvep_data(args):

        ## Load data
        ssvepdata = SSVEPLoader(args)
        args = ssvepdata.update_args()
        args = set_unseen_valid_labels(args)
        eeg_subj = ssvepdata.get_eeg_data(args.subject) # (block slide class chnl tp)
        eeg_subjs_pretrain = ssvepdata.get_eeg_data(args.subjects_pretrain) if args.is_pretrain==1 else []
        sc_template = ssvepdata.get_sc_template() # (slide class 2Nh tp)
        return eeg_subj, sc_template, eeg_subjs_pretrain, args



if __name__ == "__main__":
    main()
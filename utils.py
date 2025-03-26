import torch
import numpy as np
import random
import csv
from filelock import Timeout, FileLock
import os
import io
from torchsummary import summary
import matplotlib.pyplot as plt

import pickle
import torch.nn as nn

from Algos.GZSL import *
from Algos.Sine2SSVEP import *
from Algos.LinearModels import *



def visualize_weight_harmonic(model, args):
    conv_spatial = model.cnn_expand[1].weight.data.cpu().detach().numpy() # Nexpand(out) * Nexpand(in) * N2h * 1
    path = './Outputs/Results/Visualize_harmonic/'
    ensure_path(path)
    for chii in range(conv_spatial.shape[0]):
        kernel = conv_spatial[chii,:,:,:].squeeze().transpose()
        kernel = (kernel - np.min(kernel)) / (np.max(kernel) - np.min(kernel)+1e-8)
        plt.figure(figsize=(8, 5))
        im = plt.imshow(kernel, cmap='Reds')
        plt.colorbar(im)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(path, f'{args.dataset}_u{args.unseen_num}_s{args.subject}_t{args.window:.1f}_block{args.blkii[0]}_fb{args.fbii}_ch{chii}.svg'))
        # plt.show()
    




def create_file_detail(args):
    ## Save templates of all classes , use indices in analysis to separate (seen&unseen)
    data = {
        'args': args,
        'recon_templates': np.zeros((args.block_num, args.class_num, args.fb_num, args.chnl_num, args.timepoint_num, )),
        'true_templates': np.zeros((args.block_num, args.class_num, args.fb_num, args.chnl_num, args.timepoint_num, )),
        'test_results': np.zeros((args.block_num*args.class_num, 2)), # (true, predict) * blk*cls,
        'confusemat': np.zeros((args.class_num, args.class_num)),
        'acc': 0.0,
    }
    path_suffix = './Outputs/Results/Details'
    ensure_path(path_suffix)
    fn_detail = f"{args.dataset}-u{args.unseen_num}-t{args.window:.1f}-s{args.subject}.pickle"
    fn_detail = os.path.join(path_suffix, fn_detail)
    file = open(fn_detail, 'wb')
    pickle.dump(data, file)
    file.close()
    args.fn_detail = fn_detail

    return args

def update_template_in_detail(args, seenset, unseenset, ssvep_templates_obtained, fbii):
    # ## Save templates new
    # true_unseen_templates = np.mean(unseenset['data'][:,0,:,:,:], 0) # unseen*chnl*tp
    # recon_unseen_templates = ssvep_templates_obtained[unseenset['label']]
    # file = open(args.fn_detail, 'rb')
    # data = pickle.load(file)
    # data['true_templates'][:, fbii, :, :] = true_unseen_templates
    # data['recon_templates'][:, fbii, :, :] = recon_unseen_templates
    # file.close()

    true_templates = np.zeros((args.class_num, args.chnl_num, args.timepoint_num))
    true_templates[seenset['label'],:,:] = np.mean(seenset['data'][:,0,:,:,:], 0)
    true_templates[unseenset['label'],:,:] = np.mean(unseenset['data'][:,0,:,:,:], 0)
    file = open(args.fn_detail, 'rb')
    data = pickle.load(file)
    file.close()
    data['true_templates'][args.blkii, :, fbii, :, :] = true_templates
    data['recon_templates'][args.blkii, :, fbii, :, :] = ssvep_templates_obtained
    
    file = open(args.fn_detail, 'wb')
    pickle.dump(data, file)
    file.close()

def find_label_bm2bt():
    freqs_benchmark = [ 8. , 9. , 10. , 11. , 12. , 13. , 14. , 15. , 8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2, 8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4, 8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6, 8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8]
    freqs_beta = [ 8.6, 8.8, 9. , 9.2, 9.4, 9.6, 9.8, 10. , 10.2, 10.4, 10.6, 10.8, 11. , 11.2, 11.4, 11.6, 11.8, 12. , 12.2, 12.4, 12.6, 12.8, 13. , 13.2, 13.4, 13.6, 13.8, 14. , 14.2, 14.4, 14.6, 14.8, 15. , 15.2, 15.4, 15.6, 15.8, 8. , 8.2, 8.4]

    labels_bm2bt = []
    for idx_bm in range(len(freqs_benchmark)):
        freq_bm = freqs_benchmark[idx_bm]
        for idx_bt in range(len(freqs_beta)):
            freq_bt = freqs_beta[idx_bt]
            if abs(freq_bm - freq_bt) < 0.01:
                labels_bm2bt.append(idx_bt)
                break
        
    return labels_bm2bt


def init_weights(m):
    """Initialize NN model weights"""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01) # bug: uncomment before tformer
    elif isinstance(m, (nn.Conv2d, nn.Conv1d, nn.LSTM, nn.MultiheadAttention)):
        # torch.nn.init.xavier_uniform(m.weight.data) # not work with LSTM
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)


def get_optimizer(params, args, lr):
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True) # nesterov: follow playbook
    else:
        raise ValueError("args.opt not found!")
    return optimizer


def cal_corr(out, y, eps=1e-6):
    """
    Calculate correlation<regressed template, true/mean template>
    Parameters
    ----------
    out - ndarray (samples,chnl,tp): predicted mean templates
    y - ndarray (samples, chnl, tp): true mean templates

    Returns
    ----------
    mean correlation(out,y) of all samples

    Implementation idea: (batch*chnl, 1, tp). torch.matmul
    """
    out = torch.reshape(out, (-1, 1, out.shape[-1])) # batch*chnl, 1, tp
    y = torch.reshape(y, (-1, 1, y.shape[-1])) # batch*chnl, 1, tp
    out = out - torch.mean(out, dim=-1, keepdim=True)
    y = y - torch.mean(y, dim=-1, keepdim=True)

    corrs = torch.sum(out*y, dim=-1) / (torch.norm(out, dim=-1) * torch.norm(y, dim=-1) + eps)
    # corrs_valid = corrs[abs(corrs)<1] if is_constrain else corrs # segments might be zeros --> std=0 --> inf / nan
    corrs_valid = corrs
    corr = torch.mean(corrs_valid)
    if torch.isnan(corr):
        raise ValueError('corr is nan!')
    
    return corr

# def cosine_embedding_loss(out, y):
#     return 1 - cal_corr(out, y)


class PearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super(PearsonCorrelationLoss, self).__init__()

    def forward(self, out, y, eps=1e-6):
        corr_mean = cal_corr(out, y, eps)
        return (1-corr_mean)
    

def get_criterion(loss):
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'corr':
        criterion = PearsonCorrelationLoss()
    else:
        raise ValueError("args.loss not found!")
    return criterion


def random_fix(seed, is_cudnn_fixed=True):
    """Set fixed random seed for reproduction"""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = is_cudnn_fixed
    np.random.seed(seed)
    random.seed(seed)


def select_model(args, subspace_num):
    
    ## Classification model for regression
    ## Linear
    if args.model == 'lrchnl_c':
        model = LinearRegChnl(args.harmonic_x2, args.chnl_num).to(args.device)
    elif args.model == 'srrnet':
        model = SRRNet(N2h=args.harmonic_x2, Nel=subspace_num, dp=args.dropout).to(args.device)
    elif args.model == 'rescnn':
        model = RegResCNN(N2h=args.harmonic_x2, Nel=subspace_num, dp=args.dropout).to(args.device)
    else:
        raise ValueError("Model not found!")

    return model


def print_args(args):
    """Record progress status into txt"""
    file = open('./Outputs/Results/results.txt', 'a')
    args_dict = vars(args)
    content = ''
    for arg in args_dict:
        content += f"{arg}: {args_dict[arg]} \n"
    content += '\n\n'
    file.write(content)
    file.close()


def get_trainable_parameter_num(*models):
    total_params = 0
    for model in models:
        total_params += sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def get_model_summary(model, input_size):
    model_state = summary(model, tuple(input_size))
    return str(model_state)


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


def name_record_files(args):
    goal = 'CSV' # change as task changes
    unseen_num = len(args.label_unseen)
    if args.model in ['trca', 'tdca']:
        condition = f"{str(args.dataset)}-{str(args.model)}-t{args.window}s-Nb{args.trad_trainblocks}-fb{args.fb_num}"
    elif args.model[-4:] == 'gzsl':
        condition = f"{str(args.dataset)}-u{unseen_num}-t{args.window}s-{str(args.model)}-fb{args.fb_num}"
    else:
        condition = f"{str(args.dataset)}-u{unseen_num}-t{args.window}s-{str(args.spatialfilter)}-{str(args.model)}-fb{args.fb_num}"
    file_format = '.csv'
    path_folder = os.path.join("./Outputs/Results", goal)
    ensure_path(path_folder)
    path_csv = os.path.join(path_folder, condition+f"-{args.timenow}"+file_format)
    path_pkl = os.path.join(path_folder, condition+'.pickle')
    return path_csv, path_pkl


def init_recordings(csv_path, pkl_path, args):
    ## Initialize pickle file
    confusemat_subjs = np.zeros((args.subjects, args.class_num, args.class_num))
    with open(pkl_path, 'wb') as file:
        pickle.dump(confusemat_subjs, file)
    
    # Initialize Acc=0 for each subject (subjects * rows in total)
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for _ in range(args.subjects):
            writer.writerow([0,0,0,0,0,0,0]) # test val
        writer.writerow([])
        writer.writerow(['Time', args.timenow])
        writer.writerow(['Acc,\tTrainloss,\tValidloss,\tUnseenloss,\tTraincorr,\tValidcorr,\tUnseencorr'])
        writer.writerow(['Label unseen', args.label_unseen])
        writer.writerow(['Label validate', args.label_valid])

        # print args info
        args_dict = vars(args)
        for arg in args_dict:
            writer.writerow([arg, args_dict[arg]])


def update_csv(csv_path, pkl_path, acc, losses, corrs, confusemat, args, timeout=30):
    folder_path = 'Tmp'
    ensure_path(folder_path)
    lock_path = os.path.join(folder_path, 'file.lock')
    lock = FileLock(lock_path, timeout=timeout)
    try:
        with lock:
            # read & write results
            with open(csv_path, 'r', newline='') as file:
                reader = csv.reader(file)
                data = list(reader)

            data[args.subject] = [acc, losses[0], losses[1], losses[2], corrs[0], corrs[1], corrs[2]]

            with open(csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(data)

            with open(pkl_path, 'rb') as file:
                confusemat_subjs = pickle.load(file)
            confusemat_subjs[args.subject] = confusemat
            with open(pkl_path, 'wb') as file:
                pickle.dump(confusemat_subjs, file)
                
    except Timeout:
        print(f"CSV file occupied over {timeout} seconds (unnormal), please check the code.")

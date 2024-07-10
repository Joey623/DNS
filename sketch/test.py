import torchvision.transforms as transforms
from data_loader import PKUdata, TestData
from data_manager import *
from eval_metrics import eval_pku
from utils import *
import argparse
from model import embed_net
import torch.backends.cudnn as cudnn
from loss import PairCircle, DCL, MSEL, CenterLoss, OriTripletLoss
from re_rank import random_walk, k_reciprocal
import logging
import math
from torch.cuda.amp import autocast, GradScaler
import os
import sys
import tempfile
import torch
from torch.autograd import Variable
import torch.utils.data as data
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
# each changes
parser.add_argument('--noticed', default='test', type=str)

parser.add_argument('--dataset', default='pku', help='dataset name: regdb or sysu')
parser.add_argument('--lr', default=0.0009, type=float, help='learning rate, 0.00035 for adam, 0.0007for adamw')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=1000, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--num_pos', default=1, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--test-batch', default=8, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--margin', default=0.5, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial for PKU dataset')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--gpu', default='0,1,2,3', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--pool_dim', default=2048)
parser.add_argument('--decay_step', default=16)
parser.add_argument('--warm_up_epoch', default=8, type=int)
parser.add_argument('--max_epoch', default=100)
parser.add_argument('--rerank', default='no', type=str)
parser.add_argument('--dim', default=2048, type=int)


args = parser.parse_args(args=[])
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def extract_gall_feat(gallery_loader):
    with torch.no_grad():
        model.eval()
        # print('Extracting gallery features...')
        ptr = 0
        gallery_feats = np.zeros((ngall, args.dim))
        gallery_global_feats = np.zeros((ngall, args.dim))
        with torch.no_grad():
            for idx, (img, _) in enumerate(gallery_loader):
                img = Variable(img.to(device))
                global_feat, feat = model(img, img, modal=test_mode[0])
                batch_num = img.size(0)
                gallery_feats[ptr:ptr + batch_num, :] = feat.cpu().numpy()
                gallery_global_feats[ptr:ptr + batch_num, :] = global_feat.cpu().numpy()
                ptr = ptr + batch_num
    return gallery_global_feats, gallery_feats


def extract_query_feat(query_loader):
    with torch.no_grad():
        model.eval()
        ptr = 0
        query_feats = np.zeros((nquery, args.dim))
        query_global_feats = np.zeros((nquery, args.dim))
        with torch.no_grad():
            for idx, (img, _) in enumerate(query_loader):
                img = Variable(img.to(device))
                batch_num = img.size(0)
                global_feat, feat = model(img, img, modal=test_mode[1])
                query_feats[ptr:ptr + batch_num, :] = feat.cpu().numpy()
                query_global_feats[ptr:ptr + batch_num, :] = global_feat.cpu().numpy()
                ptr = ptr + batch_num
    return query_global_feats, query_feats


dataset = args.dataset
if dataset == 'pku':
    data_path = './PKU_sketch/'
    num_classes = 150
    test_mode = [1, 2]

cudnn.benchmark = True
print('==> Building model......')

model = embed_net(class_num=num_classes)
model.to(device)
print('==> Testing......')
# define transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize])

if args.dataset == 'pku':
    for trial in range(10):
        test_trial = trial + 1
        model_path = args.model_path + args.dataset + '/' + 'pku_p8_n1_lr_0.0009_seed_0_trial_{}_best.pth'.format(
            test_trial)
        checkpoint = torch.load(model_path)
        print('==> best epoch', checkpoint['epoch'])
        model.load_state_dict(checkpoint['net'])
        query_img, query_label = process_test_pku(data_path, modal='sketch')
        gall_img, gall_label = process_test_pku(data_path, modal='visible')

        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

        gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        nquery = len(query_label)
        ngall = len(gall_label)

        query_feat, query_feat_att = extract_query_feat(query_loader)
        gall_feat, gall_feat_att = extract_gall_feat(gall_loader)


        distmat = -np.matmul(query_feat, np.transpose(gall_feat))
        distmat_att = -np.matmul(query_feat_att, np.transpose(gall_feat_att))

        cmc, mAP, mINP = eval_pku(distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att = eval_pku(distmat_att, query_label, gall_label)

        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP
            all_cmc_att = cmc_att
            all_mAP_att = mAP_att
            all_mINP_att = mINP_att

        else:
            all_cmc += cmc
            all_mAP += mAP
            all_mINP += mINP
            all_cmc_att += cmc_att
            all_mAP_att += mAP_att
            all_mINP_att += mINP_att

        print('Test Trial: {}, Sketch to Visible'.format(test_trial))

        print(
            'original feature:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'feature after BN:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))

all_cmc = all_cmc / 10
all_mAP = all_mAP / 10
all_mINP = all_mINP / 10
all_cmc_att = all_cmc_att / 10
all_mAP_att = all_mAP_att / 10
all_mINP_att = all_mINP_att / 10
print('All Average:')
print(
    'original feature:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        all_cmc[0], all_cmc[4], all_cmc[9], all_cmc[19], all_mAP, all_mINP))
print(
    'feature after BN:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        all_cmc_att[0], all_cmc_att[4], all_cmc_att[9], all_cmc_att[19], all_mAP_att, all_mINP_att))

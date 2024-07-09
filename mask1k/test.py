from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from data_loader import *
from data_manager import *
from eval_metrics import eval_mask1k
from utils import *
from model import embed_net

# from corruptions import corruption_transform

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
# each changes
parser.add_argument('--noticed', default='VI-ReID-test', type=str)

parser.add_argument('--dataset', default='mask1k', help='dataset name: regdb or sysu')
# 0.00068
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam, 0.0007for adamw')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=1000, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=6, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--margin', default=0.5, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--resume', '-r', default='mask1k_styleABCDEF_sgd_p6_n4_lr_0.1_seed_0_best.pth', type=str,
                    help='resume from checkpoint')
parser.add_argument('--train_style', default='ABCDEF', type=str,
                    help='using which styles as the trainset, can be any combination of A-F')
parser.add_argument('--test_style', default='ABCDEF', type=str,
                    help='using which styles as the testset, can be any combination of A-F')

parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')

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
parser.add_argument('--tvsearch', default=0, type=int, help='1:visible to infrared, 0:infrared to visible')
parser.add_argument('--tta', default=False, type=bool)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
set_seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def extract_gall_feat(gallery_loader):
    with torch.no_grad():
        model.eval()
        # print('Extracting gallery features...')
        start_time = time.time()
        ptr = 0
        gallery_feats = np.zeros((ngall, args.dim))
        gallery_global_feats = np.zeros((ngall, args.dim))
        with torch.no_grad():
            for idx, (img, _) in enumerate(gallery_loader):
                img = Variable(img.to(device))
                global_feat, feat = model(img, img, modal=test_mode[0])
                if args.tta:
                    global_feat_tta, feat_tta = model(torch.flip(img, dims=[3]), torch.flip(img, dims=[3]), modal=test_mode[0])
                    global_feat += global_feat_tta
                    feat += feat_tta
                batch_num = img.size(0)
                gallery_feats[ptr:ptr + batch_num, :] = feat.cpu().numpy()
                gallery_global_feats[ptr:ptr + batch_num, :] = global_feat.cpu().numpy()
                ptr = ptr + batch_num
        duration = time.time() - start_time
    # print('Extracting time: {}s'.format(int(round(duration))))
    return gallery_global_feats, gallery_feats


def extract_query_feat(query_loader):
    with torch.no_grad():
        model.eval()
        ptr = 0
        query_feats = np.zeros((nquery, args.dim))
        query_global_feats = np.zeros((nquery, args.dim))
        if len(args.test_style) == 1:
            with torch.no_grad():
                for idx, (img, _) in enumerate(query_loader):
                    img = Variable(img.to(device))
                    batch_num = img.size(0)
                    global_feat, feat = model(img, img, modal=test_mode[1])
                    if args.tta:
                        global_feat_tta, feat_tta = model(torch.flip(img, dims=[3]), torch.flip(img, dims=[3]), modal=test_mode[1])
                        global_feat += global_feat_tta
                        feat += feat_tta
                    query_feats[ptr:ptr + batch_num, :] = feat.cpu().numpy()
                    query_global_feats[ptr:ptr + batch_num, :] = global_feat.cpu().numpy()
                    ptr = ptr + batch_num
        else:
            with torch.no_grad():
                for idx, (img, _, _) in enumerate(query_loader):
                    img = Variable(img.to(device))
                    batch_num = img.size(0)
                    global_feat, feat = model(img, img, modal=test_mode[1])
                    if args.tta:
                        global_feat_tta, feat_tta = model(torch.flip(img, dims=[3]), torch.flip(img, dims=[3]), modal=test_mode[1])
                        global_feat += global_feat_tta
                        feat += feat_tta
                    query_feats[ptr:ptr + batch_num, :] = feat.cpu().numpy()
                    query_global_feats[ptr:ptr + batch_num, :] = global_feat.cpu().numpy()
                    ptr = ptr + batch_num
    return query_global_feats, query_feats

dataset = args.dataset
data_path = "../Market-Sketch-1K/"
test_mode = [1, 2]
# args, unparsed = parser.parse_known_args()
trainset = Mask1kData_single(data_path, args.train_style, args)
color_pos, sketch_pos = GenIdx(trainset.train_color_label, trainset.train_sketch_label)

print('==> Testing......')
# define transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize])


end = time.time()
# only one sketch style as query
if len(args.test_style) == 1:
    query_img, query_label = process_test_mask1k_single(data_path, test_style=args.test_style)
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
else:
    query_img, query_label, query_style = process_test_market_ensemble(data_path, test_style=args.test_style)
    queryset = TestData_ensemble(query_img, query_label, query_style, transform=transform_test,
                                 img_size=(args.img_w, args.img_h))

gall_img, gall_label = process_test_market(data_path, modal='photo')

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('data init success!')

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  sketch  | {:5d} | {:8d}'.format(n_class, len(trainset.train_sketch_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))


cudnn.benchmark = True
# cudnn.deterministic = True

print('==> Building model......')

model = embed_net(class_num=n_class)
model.to(device)


if len(args.resume) > 0:
    model_path = args.model_path + args.dataset + '/' + args.resume
    print('==> Loading weights from checkpoint......')
    checkpoint = torch.load(model_path)
    print('==> best epoch', checkpoint['epoch'])
    model.load_state_dict(checkpoint['net'])
else:
    print('==> no checkpoint found at {}'.format(args.resume))

query_feat, query_feat_att = extract_query_feat(query_loader)
gall_feat, gall_feat_att = extract_gall_feat(gall_loader)

distmat = -np.matmul(query_feat, np.transpose(gall_feat))
distmat_att = -np.matmul(query_feat_att, np.transpose(gall_feat_att))

cmc, mAP, mINP = eval_mask1k(distmat, query_label, gall_label)
cmc_att, mAP_att, mINP_att = eval_mask1k(distmat_att, query_label, gall_label)

print(
    'original feature:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
print(
    'feature after BN:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
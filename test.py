import torchvision
from networks.fdrnet import FDRNet
from datasets.sbu_dataset import SBUDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# ckpt_path = 'ckpt/istd_epoch_010.pt'
ckpt_path = 'ckpt/sbu_epoch_010.pt'
data_root = '/home/lzhu68/hd2t/Dataset/SBU-shadow/SBU-Test'
# data_root = '/home/lzhu68/hd2t/Dataset/UCF/GouSplit'
# data_root = '/home/lzhu68/hd2t/Dataset/ISTD_Dataset/sbu_struct/test'
save_dir = 'test/raw'

os.makedirs(save_dir, exist_ok=True)

model = FDRNet(backbone='efficientnet-b3',
               proj_planes=16,
               pred_planes=32,
               use_pretrained=True,
               fix_backbone=False,
               has_se=False,
               dropout_2d=0,
               normalize=True,
               mu_init=0.4,
               reweight_mode='manual')
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt['net'])
# model.fr.set_mu(0.4)
model.cuda()
model.eval()


test_dataset = SBUDataset(data_root=data_root,
                    img_dirs=['ShadowImages'],
                    mask_dir='ShadowMasks',
                    augmentation=True,
                    phase='test',
                    normalize=False)
# print(test_dataset[0])
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)

with torch.no_grad():
    for data in tqdm(test_loader):
        im = data['ShadowImages_input'].cuda()
        im_name = data['im_name'][0]
        save_path = os.path.join(save_dir, im_name)
        gt = data['gt'][0]
        pred = torch.sigmoid(model(im)['logit'].cpu())[0]
        imgrid = torchvision.utils.save_image([im.cpu()[0], pred.expand_as(im[0]), gt.expand_as(im[0])], fp=save_path, nrow=3, padding=0)




from utils.evaluation import evaluate

im_grid_dir = 'test/raw'
pos_err, neg_err, ber, acc, df = evaluate(im_grid_dir, pred_id=1, gt_id=2, nimg=3, nrow=3)
print(f'\t BER: {ber:.2f}, pErr: {pos_err:.2f}, nErr: {neg_err:.2f}, acc:{acc:.4f}')
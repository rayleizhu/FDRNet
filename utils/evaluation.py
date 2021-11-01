import numpy as np
from collections import OrderedDict
import pandas as pd
import os
from tqdm import tqdm
import cv2
from utils.misc import split_np_imgrid, get_np_imgrid
import pydensecrf.densecrf as dcrf


def cal_ber(tn, tp, fn, fp):
    return  0.5*(fp/(tn+fp) + fn/(fn+tp))

def cal_acc(tn, tp, fn, fp):
    return (tp + tn) / (tp + tn + fp + fn)


def get_binary_classification_metrics(pred, gt, threshold=None):
    if threshold is not None:
        gt = (gt > threshold)
        pred = (pred > threshold)
    TP = np.logical_and(gt, pred).sum()
    TN = np.logical_and(np.logical_not(gt), np.logical_not(pred)).sum()
    FN = np.logical_and(gt, np.logical_not(pred)).sum()
    FP = np.logical_and(np.logical_not(gt), pred).sum()
    BER = cal_ber(TN, TP, FN, FP)
    ACC = cal_acc(TN, TP, FN, FP)
    return OrderedDict( [('TP', TP),
                        ('TN', TN),
                        ('FP', FP),
                        ('FN', FN),
                        ('BER', BER),
                        ('ACC', ACC)]
                      )


def evaluate(res_root, pred_id, gt_id, nimg, nrow):
    img_names  = os.listdir(res_root)
    score_dict = OrderedDict()

    for img_name in tqdm(img_names, disable=False):
        im_grid_path = os.path.join(res_root, img_name)
        im_grid = cv2.imread(im_grid_path)
        ims = split_np_imgrid(im_grid, nimg, nrow)
        pred = ims[pred_id]
        gt = ims[gt_id]
        score_dict[img_name] = get_binary_classification_metrics(pred,
                                                                 gt,
                                                                 125)
            
    df = pd.DataFrame(score_dict)
    df['ave'] = df.mean(axis=1)

    tn = df['ave']['TN']
    tp = df['ave']['TP']
    fn = df['ave']['FN']
    fp = df['ave']['FP']

    pos_err = (1 - tp / (tp + fn)) * 100
    neg_err = (1 - tn / (tn + fp)) * 100
    ber = (pos_err + neg_err) / 2
    acc = (tn + tp) / (tn + tp + fn + fp)

    return pos_err, neg_err, ber, acc, df



def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def crf_refine(img, annos):
    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')

    

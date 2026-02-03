import tqdm
import math
import h5py
import torch
import argparse
import numpy as np
# import nibabel as nib
import matplotlib.pyplot as plt
from skimage.measure import label

from utils.utils import calculate_metric_percase



def getLargestCC(segmentation):
    labels = label(segmentation)
    # assert( labels.max() != 0 ) # assume at least 1 CC
    if labels.max() != 0:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    else:
        largestCC = segmentation
    return largestCC


def test_single_case(args, model, image, stride_xy, stride_z, patch_size, num_classes):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
        
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))

    
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    
    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    outputs, _, _ = model(test_patch, is_training=False)
                    
                    y = torch.softmax(outputs, dim=1)

                y = y.cpu().data.numpy()
                y = y[0, 1, :, :, :]
                
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] = score_map[:,xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1

    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = (score_map[0] > 0.5).astype(np.int64)
    
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        
    return label_map, score_map
    


def test_calculate_metric(args, model, val=False):
    args.root_path = args.data_path + args.dataset
    if args.dataset == "/Pancreas":
        with open(args.root_path + '/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = [args.root_path + "/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]
    elif args.dataset == "/Lung":
        with open(args.root_path + '/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = [args.root_path + "/lung_h5/" + item.replace('\n', '') + ".h5" for item in image_list]
    elif "BraTS" in args.dataset:
        if val:
            with open(args.root_path + '/data/val.list', 'r') as f:
                image_list = f.readlines()
        else:
            with open(args.root_path + '/data/test.list', 'r') as f:
                image_list = f.readlines()
        image_list = [args.root_path + "/data/" + item.replace('\n', '') + ".h5" for item in image_list]
    else:
        with open(args.root_path + '/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = [args.root_path + "/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]

    if "BraTS2019" in args.dataset:
        # stride_xy = 18
        # stride_z = 18
        stride_xy = 64
        stride_z = 64
    elif "Pancreas" in args.dataset:
        stride_xy = 16
        stride_z = 16
    elif "Lung" in args.dataset:
        stride_xy = 18
        stride_z = 4
    else:
        stride_xy = 18
        stride_z = 4

    patch_size = args.patch_size
    

    avg_dice_list = []
    avg_hd95_list = []
    avg_asd_list = []
    avg_iou_list = []
    

    ith = 0
    for image_path in image_list:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]

        label_batch = torch.from_numpy(label.astype("float32"))

        prediction, _ = test_single_case(args,  model, image, stride_xy, stride_z, patch_size, num_classes=args.num_classes)
        
        if 1:
            prediction = getLargestCC(prediction)


        dice, iou, hd95, asd = calculate_metric_percase(label_batch, prediction, thr=0.5)


        if not val:
            print(dice, iou, hd95, asd)
            
        avg_dice_list.append(dice)
        avg_hd95_list.append(hd95)
        avg_iou_list.append(iou)
        avg_asd_list.append(asd)
        
        
        ith += 1
        
    avg_dice = np.mean(avg_dice_list)
    avg_hd95 = np.mean(avg_hd95_list)
    avg_iou = np.mean(avg_iou_list)
    avg_asd = np.mean(avg_asd_list)      
    
    if not val:
        print("avg_dice: ", avg_dice)
        print("avg_iou: ", avg_iou)
        print("avg_hd95: ", avg_hd95)
        print("avg_asd: ", avg_asd)
        
    return avg_dice, avg_hd95
        


if __name__ == '__main__':
    
    from model.Semi_MoE import VNet_MoE

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='/opt/data/private/data_3D',
                        help='Name of Experiment')
    
    parser.add_argument('--dataset', type=str, default='/2018LA_Seg_Training Set',
                        help='Name of Experiment')
    # parser.add_argument('--dataset', type=str, default='/Pancreas',
    #                     help='Name of Experiment')
    # parser.add_argument('--dataset', type=str, default='/BraTS2019',
    #                 help='Name of Experiment')
    # parser.add_argument('--dataset', type=str, default='/Lung',
    #                     help='Name of Experiment')

    parser.add_argument('--in_channels', type=int, default=1,
                        help='output channel of network')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='output channel of network')

    parser.add_argument('--detail', type=int, default=1,
                        help='print metrics for every samples?')
    parser.add_argument('--nms', type=int, default=1,
                        help='apply NMS post-procssing?')

    parser.add_argument('--patch_size', type=list,  default=[112, 112, 80],
                        help='patch size of network input')
    
    # parser.add_argument('--patch_size', type=list, default=[96, 96, 96],
    #                     help='patch size of network input')

    parser.add_argument('--model_path', type=str,
                        default="./Results/..pth",
                        help='model weight path')

    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()


    # init Seg model
    model = VNet_MoE(args=args, n_channels=args.in_channels, n_classes=args.num_classes).to(args.device).eval()

    # load checkpoint
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint)

    print("init weight from {}".format(args.model_path))
    metric = test_calculate_metric(args, model, val=False, save_path=None)
    # print(metric)
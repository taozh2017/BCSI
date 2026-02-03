import os
import cv2
import torch
import random
import logging
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from prediction import test_calculate_metric

from utils.mix_up import generate_mask_3D, get_cut_mask
from utils.losses import DiceLoss

from model.Semi_MoE import VNet_MoE


# from skimage.measure import label
# def LargestCC_pancreas(segmentation):
#     N = segmentation.shape[0]
#     batch_list = []
#     for n in range(N):
#         n_prob = segmentation[n].detach().cpu().numpy()
#         labels = label(n_prob)
#         if labels.max() != 0:
#             largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
#         else:
#             largestCC = n_prob
#         batch_list.append(largestCC)
    
#     return torch.Tensor(batch_list).cuda()
    
# def get_cut_mask(out, thres=0.5, nms=1):
#     probs = F.softmax(out, 1)
#     masks = (probs >= thres).type(torch.int64)
#     masks = masks[:, 1, :, :, :].contiguous()
#     if nms == 1:
#         masks = LargestCC_pancreas(masks)
#     return masks


def get_cut_mask(P, temperature=0.1):
    """
    Inputs:
        P: [B, C, D, H, W] - logits (before softmax)
        temperature: float - lower means harder pseudo labels
    Returns:
        pseudo_labels: [B, D, H, W] - class index with highest sharpened prob
    """
    P = torch.softmax(P, dim=1)  # Convert logits to probability
    P_sharpen = P ** (1 / temperature)
    P_sharpen = P_sharpen / torch.sum(P_sharpen, dim=1, keepdim=True)
    pseudo_labels = torch.argmax(P_sharpen, dim=1)
    return pseudo_labels

class PolyWarmRestartScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, base_lr, max_iters, power=0.9, warm_restart_iters=5000, last_epoch=-1):
        self.base_lr = base_lr
        self.max_iters = max_iters
        self.power = power
        self.warm_restart_iters = warm_restart_iters
        self.current_cycle_start = 0
        super(PolyWarmRestartScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch - self.current_cycle_start
        if t >= self.warm_restart_iters:
            self.current_cycle_start = self.last_epoch
            t = 0
        factor = (1 - t / self.warm_restart_iters) ** self.power
        return [self.base_lr * factor for _ in self.base_lrs]
    

def one_hot_encoder(input_tensor, n_classes):
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


class Trainer(nn.Module):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.best_performance = 0.0
        
        self.args = args

        # init Seg model
        self.model = VNet_MoE(args=args, n_channels=args.in_channels, n_classes=args.num_classes).to(args.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.00001)
        
        self.scheduler = PolyWarmRestartScheduler(
            self.optimizer,
            base_lr=args.base_lr,
            max_iters=args.max_iterations,
            power=0.9,
            warm_restart_iters=8000
        )

        self.dice_loss = DiceLoss(args.num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self.pixel_ce_loss = nn.CrossEntropyLoss(reduction='none')
        
        
    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.args.consistency * self.sigmoid_rampup(epoch, self.args.consistency_rampup)
    
    def get_entropy_map(self, p, softmax=True):
        if softmax:
            p = torch.softmax(p, dim=1)
        ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
        return ent_map
    
    def weit_loss(self, pred, mask, weit, weit_coef=2, smooth=1e-6):
        weit = weit * weit_coef + 1
        mask_one_hot = F.one_hot(mask.long(), num_classes=2).permute(0, 4,  1, 2, 3).float()

        wbce = F.binary_cross_entropy_with_logits(pred, mask_one_hot, reduction='none')
        wbce = (weit * wbce).mean()
        
        # IOU
        pred_sig = torch.sigmoid(pred)
        inter = (pred_sig * mask_one_hot * weit).sum(dim=(2, 3, 4))
        union = (pred_sig * weit).sum(dim=(2, 3, 4)) + (mask_one_hot * weit).sum(dim=(2, 3, 4)) - inter
        wiou = 1 - ((inter + smooth)/(union + smooth)).mean(dim=1)
        
        if wiou.mean() < 0:
            print("Error", wiou.mean())
        
        loss = wbce + wiou.mean()
        return loss
    
    def train(self, sampled_batch, iter_num, snapshot_path):
        
        weak_aug_volume_batch, strong_aug_volume_batch = sampled_batch['weak_aug'], sampled_batch['strong_aug']
        
        weak_lab_img = weak_aug_volume_batch["image"][:self.args.labeled_bs].to(self.args.device)
        weak_lab_label = weak_aug_volume_batch["label"][:self.args.labeled_bs].to(self.args.device)
        weak_unlab_img = weak_aug_volume_batch["image"][self.args.labeled_bs:].to(self.args.device)
        
        strong_lab_img = strong_aug_volume_batch["image"][:self.args.labeled_bs].to(self.args.device)
        strong_lab_label = strong_aug_volume_batch["label"][:self.args.labeled_bs].to(self.args.device)
        strong_unlab_img = strong_aug_volume_batch["image"][self.args.labeled_bs:].to(self.args.device)
        
        # copy paste 
        mask, _ = generate_mask_3D(weak_lab_img, mask_ratio=2/3) # 2/3 mask
        
        # lab unlab copy paste
        weak_lab_strong_unlab_img = weak_lab_img * mask + strong_unlab_img  * (1 - mask)
        weak_unlab_strong_lab_img = weak_unlab_img * mask + strong_lab_img  * (1 - mask)
        lab_unlab_cp_img = torch.concat([weak_lab_strong_unlab_img, weak_unlab_strong_lab_img], dim=0)
        
        # forward
        weak_pred, weak_features, weak_mask = self.model(torch.concat([weak_lab_img, weak_unlab_img], dim=0), is_training=True)
        strong_pred, strong_features, strong_mask = self.model(torch.concat([strong_lab_img, strong_unlab_img], dim=0), is_training=True)
        mixed_pred, mixed_features, mixed_mask = self.model(lab_unlab_cp_img, is_training=True)
        
        weak_uncertainty_map = self.get_entropy_map(weak_pred)
        strong_uncertainty_map = self.get_entropy_map(strong_pred)
        mixed_uncertainty_map = self.get_entropy_map(mixed_pred)
        
        # pseudo map
        unlab_pseudo = get_cut_mask(weak_pred[self.args.labeled_bs:]).long().clone().detach()
        
        # restore mixed img pred
        mixed_lab_pred = mixed_pred[:self.args.labeled_bs] * mask + mixed_pred[self.args.labeled_bs:] * (1 - mask)
        mixed_lab_uncertainty_map = mixed_uncertainty_map[:self.args.labeled_bs] * mask + mixed_uncertainty_map[self.args.labeled_bs:] * (1 - mask)
        
        mixed_unlab_pred = mixed_pred[:self.args.labeled_bs] * (1 - mask) + mixed_pred[self.args.labeled_bs:] * mask 
        mixed_unlab_uncertainty_map = mixed_uncertainty_map[:self.args.labeled_bs] * (1 - mask) + mixed_uncertainty_map[self.args.labeled_bs:] * mask 
        
        # sup loss
        sup_loss = 0
        sup_loss += self.weit_loss(weak_pred[:self.args.labeled_bs], weak_lab_label, weak_uncertainty_map[:self.args.labeled_bs])
        sup_loss += self.weit_loss(strong_pred[:self.args.labeled_bs], strong_lab_label, strong_uncertainty_map[:self.args.labeled_bs])
        sup_loss += self.weit_loss(mixed_lab_pred, weak_lab_label, mixed_lab_uncertainty_map)
        
        # # unsup loss
        unsup_loss = 0
        unsup_loss += self.weit_loss(strong_pred[self.args.labeled_bs:], unlab_pseudo, strong_uncertainty_map[:self.args.labeled_bs])
        unsup_loss += self.weit_loss(mixed_unlab_pred, unlab_pseudo, mixed_unlab_uncertainty_map)
        
        # cons_loss 
        cons_loss = 0
        mixed_pred_soft = torch.softmax(torch.concat([mixed_lab_pred, mixed_unlab_pred], dim=0), dim=1)
        cons_loss += F.mse_loss(mixed_pred_soft, torch.softmax(strong_pred, dim=1))
        
        # total loss
        consistency_weight = self.get_current_consistency_weight(iter_num // 150)
        loss = sup_loss + cons_loss + consistency_weight * unsup_loss
        
        # save
        self.model.interpolation_save(weak_features[:self.args.labeled_bs], weak_mask[:self.args.labeled_bs], weak_lab_label, is_lab=True)
        self.model.interpolation_save(weak_features[self.args.labeled_bs:], weak_mask[self.args.labeled_bs:], unlab_pseudo, is_lab=False)

        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        
        

        logging.info('iteration %d : '
                    '  loss : %f'
                    '  lr : %f'
                    
                    % (iter_num, 
                        loss, 
                        current_lr,
                        ))

    def test(self, snapshot_path, iter_num):
        self.model.eval()
        
        dice, hd95 = test_calculate_metric(self.args, self.model,  val=True)
        
        if dice > self.best_performance:
            self.best_performance = dice
            save_best = os.path.join(snapshot_path, 'Model_iter_' + str(iter_num) + ".pth")
            torch.save(self.model.state_dict(), save_best)
                
                
        logging.info('iteration %d : '
                    '  mean_dice : %f '
                    '  mean_hd95 : %f '
                    % (iter_num, dice, hd95))

        self.model.train()


        




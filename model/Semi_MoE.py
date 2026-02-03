import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling_function(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling = 1):
        super(Upsampling_function, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

    
class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        self.num_experts = 3 
        
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)


        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        
        
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res
    

class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, up_type=0):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization, mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization, mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization, mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization, mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]
        
        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        
        if self.has_dropout:
            x9 = self.dropout(x9)
            
        out_seg = self.out_conv(x9)
        return out_seg

    
class ChannelCrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(self, query, keys):  
        """
        query: [N, D]
        keys: [N, K, D]
        """
        Q = self.q_proj(query) # [N, 1, D]
        K = self.k_proj(keys)                # [N, K, D]
        V = self.v_proj(keys)                # [N, K, D]

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [N, 1, K]
        attn = torch.softmax(attn, dim=-1)                        # [N, 1, K]
        out = torch.matmul(attn, V)                               # [N, 1, D]
        return out.squeeze(1)                                     # [N, D]
    



class VNet_MoE(nn.Module):
    def __init__(self, args, n_channels=1, n_classes=2, n_filters=16, normalization='batchnorm', has_dropout=True, has_residual=True, ):
        super(VNet_MoE, self).__init__()

        self.args = args
        self.topk = 64
        channels = [16, 32, 64, 128, 256]

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        
        self.decoder = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=0)

        self.has_dropout = has_dropout
        
        # create the queue
        self.K = 2560
        self.feature_dim = (self.args.patch_size[0] // 16) * (self.args.patch_size[1] // 16) * (self.args.patch_size[2] // 16)
        self.num_classes = 1
        
        self.register_buffer("lab_queue_feature", torch.randn(self.num_classes, self.K, self.feature_dim))
        self.lab_queue_feature = F.normalize(self.lab_queue_feature, dim=2)
        self.register_buffer("unlab_queue_feature", torch.randn(self.num_classes, self.K, self.feature_dim))
        self.unlab_queue_feature = F.normalize(self.unlab_queue_feature, dim=2)

        # # queue init.
        self.register_buffer("lab_queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))
        self.register_buffer("unlab_queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))
        
        
        self.mask_learner = nn.Sequential(
                                nn.AdaptiveAvgPool3d(1),
                                nn.Conv3d(channels[-1], channels[-1] // 4, 1),
                                nn.ReLU(),
                                nn.Conv3d(channels[-1] // 4, channels[-1], 1),
                                nn.Softmax()
                            )
        
        self.channel_attention = ChannelCrossAttention(dim=self.feature_dim)
        # self.channel_attention = MultiHeadCrossAttention(dim=self.feature_dim, num_heads=4)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, feature, is_lab=False):
        """update the queue."""
        M, D = feature.shape
        if is_lab:
            queue = self.lab_queue_feature[0]  # [K, D]
            ptr = int(self.lab_queue_ptr[0])
        else:
            queue = self.unlab_queue_feature[0]
            ptr = int(self.unlab_queue_ptr[0])
            
        end_ptr = ptr + M
        if end_ptr <= self.K:
            queue[ptr:end_ptr] = feature
        else:
            overflow = end_ptr - self.K
            queue[ptr:] = feature[:M - overflow]
            queue[:overflow] = feature[M - overflow:]
        
        new_ptr = (ptr + M) % self.K
        if is_lab:
            self.lab_queue_ptr[0] = new_ptr
        else:
            self.unlab_queue_ptr[0] = new_ptr

    
    def interpolation_save(self, feature, mask, label=None, is_lab=False):
        b, channel, h, w, d = feature.shape
        dim = h * w * d
        
        feats_flat = feature.view(b, channel, -1)  # [B, C, D*H*W]
        
        _, topk_indices = torch.topk(mask, k=min(self.topk, channel), dim=1)  # [B, k]
        selected_feats = torch.zeros(b, self.topk, dim, device=feature.device)
        for i in range(b):
            selected_feats[i] = feats_flat[i, topk_indices[i]]
        
        if label is not None:
            label_resized = F.interpolate(label.float().unsqueeze(1),  size=(h, w, d), )  # [B, 1, D, H, W]
            label_flat = label_resized.view(b, 1, -1)  # [B, 1, D*H*W]
            for i in range(b):
                sample_label = label_flat[i]  # [1, D*H*W]
                selected_feats[i] *= sample_label
        
        memory_to_store = selected_feats.view(-1, dim)  # [B*k, D*H*W]
        self._dequeue_and_enqueue(memory_to_store, is_lab)
    
    def enhance_selected_channels(self, feats, mask, memory_bank):
        B, C, D, H, W = feats.shape
        dim = D*H*W
        
        feats_flat = feats.view(B, C, -1)  # [B, C, D*H*W]
        
        _, topk_indices = torch.topk(mask, k=min(self.topk, C), dim=1)  # [B, k]
        
        selected_feats = torch.zeros(B, self.topk, dim, device=feats.device)
        for i in range(B):
            selected_feats[i] = feats_flat[i, topk_indices[i]]
        
        query_feat = selected_feats.view(-1, dim)  # [B*k, D']
        query_norm = F.normalize(query_feat, dim=-1)
        mem_norm = F.normalize(memory_bank, dim=1)  # [M, D']
        
        sim = torch.matmul(query_norm, mem_norm.T)  # [B*k, M]
        _, mem_topk_idx = torch.topk(sim, k=min(1, memory_bank.size(0)), dim=1)  # [B*k, topk_mem]
        
        selected_mem = mem_norm[mem_topk_idx]  # [B*k, topk_mem, D']
        enhanced_feat = self.channel_attention(query_feat.reshape(B, -1,  dim), selected_mem.reshape(B, -1,  dim))  # [B*k, D']
        
        enhanced_feat = enhanced_feat.view(B, self.topk, dim)
        output_feats = feats_flat.clone()
        
        for i in range(B):
            output_feats[i, topk_indices[i]] += enhanced_feat[i]
        
        return output_feats.view(B, C, D, H, W)


    def forward(self, input, is_training=True):
        
        features = self.encoder(input)
        ori_feats = features[-1].clone()
        
        mask = self.mask_learner(features[-1]).view(features[-1].size(0), -1)
        mask = torch.softmax(mask, dim=1)
        
        if is_training:
            features[-1][:self.args.labeled_bs] = self.enhance_selected_channels(features[-1][:self.args.labeled_bs], mask[:self.args.labeled_bs], self.unlab_queue_feature[0])
            features[-1][self.args.labeled_bs:] = self.enhance_selected_channels(features[-1][self.args.labeled_bs:], mask[self.args.labeled_bs:], self.lab_queue_feature[0])

        pred = self.decoder(features)
        return pred, ori_feats, mask
        

        
        

        




























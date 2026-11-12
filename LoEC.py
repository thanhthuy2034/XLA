import model.resnet as resnet

import torch
from torch import nn
import torch.nn.functional as F
import pdb

class SSP_MatchingNet(nn.Module):
    def __init__(self, backbone, refine=False):
        super(SSP_MatchingNet, self).__init__()
        backbone = resnet.__dict__[backbone](pretrained=True)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3
        self.refine = refine

    def forward(self, img_s_list, mask_s_list, img_q, mask_q, use_mask_correction, mask_k, mask_b, q_ori):
        # img_s_list len = shot, [B, 3, 400, 400]
        # mask_s_list len = shot, [B, 400, 400]
        # img_q, [B, 3, 400, 400]
        # mask_q, [B, 400, 400]

        b, h, w = img_q.shape[0], img_q.shape[2], img_q.shape[3]

        # feature maps of support images
        feature_s_list = []
        for k in range(len(img_s_list)):
            with torch.no_grad():
                s_0 = self.layer0(img_s_list[k])   # [B, 128, 100, 100]
                if self.training:
                    # LEM: RandConv support feature after block0
                    s_0 = self.feat_RCfourier_bias(s_0, b, 100, 100, 128, 128, 3)
                s_0 = self.layer1(s_0)             # [B, 256, 100, 100]
            s_0 = self.layer2(s_0)                 # [B, 512, 50, 50]
            s_0 = self.layer3(s_0)                 # [B, 1024, 50, 50]
            feature_s_list.append(s_0)
            del s_0
        # feature map of query image
        with torch.no_grad():
            q_0 = self.layer0(img_q)
            q_0 = self.layer1(q_0)
        q_0 = self.layer2(q_0)
        feature_q = self.layer3(q_0)     # [B, 1024, 50, 50]

        # foreground(target class) and background prototypes pooled from K support features
        feature_fg_list = []
        feature_bg_list = []
        supp_out_ls = []
        for k in range(len(img_s_list)):
            feature_fg = self.masked_average_pooling(feature_s_list[k], (mask_s_list[k] == 1).float())[None, :]  # [1, B, 1024]
            feature_bg = self.masked_average_pooling(feature_s_list[k], (mask_s_list[k] == 0).float())[None, :]  # [1, B, 1024]
            feature_fg_list.append(feature_fg)
            feature_bg_list.append(feature_bg)

            if self.training:
                supp_similarity_fg = F.cosine_similarity(feature_s_list[k], feature_fg.squeeze(0)[..., None, None], dim=1)  # [B, 50, 50]
                supp_similarity_bg = F.cosine_similarity(feature_s_list[k], feature_bg.squeeze(0)[..., None, None], dim=1)  # [B, 50, 50]
                supp_out = torch.cat((supp_similarity_bg[:, None, ...], supp_similarity_fg[:, None, ...]), dim=1) * 10.0   # [B, 2, 50, 50]

                supp_out = F.interpolate(supp_out, size=(h, w), mode="bilinear", align_corners=True)   # [B, 2, 400, 400]
                supp_out_ls.append(supp_out)

        # average K foreground prototypes and K background prototypes
        FP = torch.mean(torch.cat(feature_fg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)   # [B, 1024, 1, 1]
        BP = torch.mean(torch.cat(feature_bg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)   # [B, 1024, 1, 1]

        # measure the similarity of query features to fg/bg prototypes
        out_0 = self.similarity_func(feature_q, FP, BP)   # [B, 2, 50, 50]

        ##################### Self-Support Prototype (SSP) #####################
        SSFP_1, SSBP_1, ASFP_1, ASBP_1 = self.SSP_func(feature_q, out_0)
        
        FP_1 = FP * 0.5 + SSFP_1 * 0.5
        BP_1 = SSBP_1 * 0.3 + ASBP_1 * 0.7

        out_1 = self.similarity_func(feature_q, FP_1, BP_1)

        ##################### SSP Refinement #####################
        if self.refine:
            SSFP_2, SSBP_2, ASFP_2, ASBP_2 = self.SSP_func(feature_q, out_1)

            FP_2 = FP * 0.5 + SSFP_2 * 0.5
            BP_2 = SSBP_2 * 0.3 + ASBP_2 * 0.7

            FP_2 = FP * 0.5 + FP_1 * 0.2 + FP_2 * 0.3
            BP_2 = BP * 0.5 + BP_1 * 0.2 + BP_2 * 0.3

            out_2 = self.similarity_func(feature_q, FP_2, BP_2)

            out_2 = out_2 * 0.7 + out_1 * 0.3

        #out_0 = F.interpolate(out_0, size=(h, w), mode="bilinear", align_corners=True)
        out_1 = F.interpolate(out_1, size=(h, w), mode="bilinear", align_corners=True)

        if self.refine:
            out_2 = F.interpolate(out_2, size=(h, w), mode="bilinear", align_corners=True)
            out_ls = [out_2, out_1]
        else:
            out_ls = [out_1]

        if self.training:
            fg_q = self.masked_average_pooling(feature_q, (mask_q == 1).float())[None, :].squeeze(0)
            bg_q = self.masked_average_pooling(feature_q, (mask_q == 0).float())[None, :].squeeze(0)

            self_similarity_fg = F.cosine_similarity(feature_q, fg_q[..., None, None], dim=1)
            self_similarity_bg = F.cosine_similarity(feature_q, bg_q[..., None, None], dim=1)
            self_out = torch.cat((self_similarity_bg[:, None, ...], self_similarity_fg[:, None, ...]), dim=1) * 10.0

            self_out = F.interpolate(self_out, size=(h, w), mode="bilinear", align_corners=True)
            supp_out = torch.cat(supp_out_ls, 0)

            out_ls.append(self_out)
            out_ls.append(supp_out)

        out_wo_mask_correction, top_k_location_a = None, None
        # LCM
        if use_mask_correction:
            out_wo_mask_correction = out_ls[0].clone()  # for vis
            # q_ori  [1, 480, 480, 3]
            top_k_location_a, top_k_similarity_map_a = self.mask_correction(out_ls[0], q_ori)
            reshaped_top_k_similarity_map_a = F.interpolate(top_k_similarity_map_a, (480, 480), mode='nearest')
            out_ls[0][:, 1, :, :] = out_ls[0][:, 1, :, :] + out_ls[0][:, 1, :, :] * (reshaped_top_k_similarity_map_a[:, 0, :, :] - mask_b) * mask_k

        return out_ls, out_wo_mask_correction, top_k_location_a

    # LCM
    def mask_correction(self, out, q_ori, patch_size=16):
        """
        Parameters
        ----------
        out: torch.Tensor
            [B, 2, h, w]   [B, 2, 400, 400]
        q_ori:
            [B, 400, 400, 3], original RGB query image
        Returns
        -------
        top_indices:
            top_indices location (0~899)
            [B, K]
        top_k similiarity map:
            [B, K, 30, 30]
        """
        # Select the patches with the highest confidence
        b, h, w = out.shape[0], out.shape[2], out.shape[3]
        out_confidence = out[:, 1, :, :] - out[:, 0, :, :]  # fg - bg  [B, 400, 400]

        # Number of patches
        num_patches_x = h // patch_size
        num_patches_y = w // patch_size
        # [b, height, width]-> [b, num_patches, num_patches, patch_size, patch_size]
        patch_confidence = out_confidence.view(b, num_patches_x, patch_size, num_patches_y, patch_size).permute(0, 1, 3, 2, 4)  # [b, 30, 30, 16, 16]
        # Compute the mean confidence of each patch
        patch_confidence_means = patch_confidence.mean(dim=(3, 4))  # [B, 40, 40]
        # Get the indices of the K patches with the highest confidence
        K = 1
        _, top_indices = torch.topk(patch_confidence_means.view(b, -1), k=K, dim=1, largest=True, sorted=True)  # [B, K]

        # patch_similarity_map
        # [B, height, width, channels]-> [B, num_patches, num_patches, patch_size, patch_size, channels]
        reshaped_image = q_ori.reshape(b, num_patches_x, patch_size, num_patches_y, patch_size, 3).permute(0, 1, 3, 2, 4, 5)   # [B, 30, 30, 16, 16, 3]
        # Compute the mean RGB value of each patch  [B, patches_height, patches_width, channels]
        patch_means = reshaped_image.to(torch.float32).mean(dim=(3, 4))  # [B, 40, 40, 3]
        patch_means_reshaped = patch_means.view(b, -1, 3)  # [B, 1600, 3]

        diff = patch_means_reshaped.unsqueeze(1) - patch_means_reshaped.unsqueeze(2)  # [B, 1600, 1600, 3]
        euclidean_squared = torch.sum(diff ** 2, dim=-1)  # [B, 1600, 1600]
        min_dist, _ = euclidean_squared.min(dim=2, keepdim=True)  # [B, 1600, 1]
        max_dist, _ = euclidean_squared.max(dim=2, keepdim=True)
        similarity = 1 - (euclidean_squared - min_dist) / (max_dist - min_dist + 1e-8)  # [B, 1600, 1600]

        # Compute the similarity maps for the top-K high-confidence patches
        batch_indices = torch.arange(b).unsqueeze(1).expand_as(top_indices)
        top_k_similarity_map = similarity[batch_indices, top_indices, :].view(b, K, 30, 30)  # [B, K, 40, 40]

        return top_indices, top_k_similarity_map



    def SSP_func(self, feature_q, out):
        # feature_q [B,1024,50,50]
        # out [B,2,50,50]
        bs = feature_q.shape[0]
        pred_1 = out.softmax(1)            # [B, 2, 50, 50]
        pred_1 = pred_1.view(bs, 2, -1)    # [B, 2, 2500]
        pred_fg = pred_1[:, 1]             # [B, 2500]
        pred_bg = pred_1[:, 0]             # [B, 2500]
        fg_ls = []
        bg_ls = []
        fg_local_ls = []
        bg_local_ls = []
        for epi in range(bs):
            fg_thres = 0.7 #0.9 #0.6
            bg_thres = 0.6 #0.6
            cur_feat = feature_q[epi].view(1024, -1)      # [1024,2500]
            f_h, f_w = feature_q[epi].shape[-2:]          # 50,50

            if (pred_fg[epi] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pred_fg[epi]>fg_thres)] #.mean(-1)                   # [1024, N1]
            else:
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 12).indices] #.mean(-1)   # [1024, 12]
            if (pred_bg[epi] > bg_thres).sum() > 0:
                bg_feat = cur_feat[:, (pred_bg[epi]>bg_thres)] #.mean(-1)
            else:
                bg_feat = cur_feat[:, torch.topk(pred_bg[epi], 12).indices] #.mean(-1)
            # global proto
            fg_proto = fg_feat.mean(-1)             # [1024]
            bg_proto = bg_feat.mean(-1)
            fg_ls.append(fg_proto.unsqueeze(0))
            bg_ls.append(bg_proto.unsqueeze(0))

            # local proto
            fg_feat_norm = fg_feat / torch.norm(fg_feat, 2, 0, True) # 1024, N1
            bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True) # 1024, N2
            cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True) # 1024, N3   1024, 2500

            cur_feat_norm_t = cur_feat_norm.t() # N3, 1024
            fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0 # N3, N1
            bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0 # N3, N2

            fg_sim = fg_sim.softmax(-1)     # 2500,N1
            bg_sim = bg_sim.softmax(-1)     # 2500,N2

            fg_proto_local = torch.matmul(fg_sim, fg_feat.t()) # N3, 1024
            bg_proto_local = torch.matmul(bg_sim, bg_feat.t()) # N3, 1024

            fg_proto_local = fg_proto_local.t().view(1024, f_h, f_w).unsqueeze(0) # 1024, N3    [1,1024,50,50]
            bg_proto_local = bg_proto_local.t().view(1024, f_h, f_w).unsqueeze(0) # 1024, N3

            fg_local_ls.append(fg_proto_local)
            bg_local_ls.append(bg_proto_local)

        # global proto
        new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)                # [4, 1024, 1, 1]
        new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)                # [4, 1024, 1, 1]

        # local proto
        new_fg_local = torch.cat(fg_local_ls, 0).unsqueeze(-1).unsqueeze(-1)     # [4, 1024, 50, 50, 1, 1]
        new_bg_local = torch.cat(bg_local_ls, 0)                                 # [4, 1024, 50, 50]

        return new_fg, new_bg, new_fg_local, new_bg_local

    def similarity_func(self, feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)

        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out

    def masked_average_pooling(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                         / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature

    # LEM
    def feat_RCfourier_bias(self, feat, b, h, w, in_channels, out_channels, kernel_size):   # single img
        # feat [B, 128, 100, 100]
        random_convolution = nn.Conv2d(b * in_channels, b * out_channels, kernel_size, padding=kernel_size // 2,  groups=b, bias=True).to(feat.device)
        torch.nn.init.kaiming_normal_(random_convolution.weight, mode="fan_in", nonlinearity="relu")

        feat_reshaped = feat.view(1, b * in_channels, h, w)
        feat_RC = random_convolution(feat_reshaped)
        feat_RC = feat_RC.view(b, out_channels, h, w)

        # feat and feat_RC FFT
        feat_fft = torch.fft.fft2(feat, dim=(-2, -1))
        feat_RC_fft = torch.fft.fft2(feat_RC, dim=(-2, -1))

        amp, phase = torch.abs(feat_fft), torch.angle(feat_fft)
        amp_RC, phase_RC = torch.abs(feat_RC_fft), torch.angle(feat_RC_fft)

        # RC amp + ori phase
        new_complex = amp_RC * torch.exp(1j * phase)
        # ori amp + RC phase
        # new_complex = amp * torch.exp(1j * phase_RC)

        new_feat_ifft = torch.fft.ifft2(new_complex)
        new_feat = torch.real(new_feat_ifft)

        return new_feat

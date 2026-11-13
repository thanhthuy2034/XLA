"""Model classes from notebookf7987d0818.ipynb, cells 4 and 6.
Training and dataset cells are never executed by the demo.
"""
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet50_LoEC(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50_LoEC, self).__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet50(weights=weights)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3

        # 1. Dilation cho layer3 theo chuẩn FSS/SSP
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation = (2, 2)
                m.padding = (2, 2)
                m.stride = (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        # 2. Bỏ ReLU cuối cùng của layer3 (Mục 4.2 bài báo: 'discard the last backbone stage and the last ReLU')
        last_block = self.layer3[-1]
        def _forward_no_last_relu(block_self, x):
            identity = x
            out = block_self.relu(block_self.bn1(block_self.conv1(x)))
            out = block_self.relu(block_self.bn2(block_self.conv2(out)))
            out = block_self.bn3(block_self.conv3(out))
            if block_self.downsample is not None:
                identity = block_self.downsample(x)
            out += identity
            return out  # Không qua ReLU cuối để giữ toàn bộ miền giá trị cho Cosine Similarity
        last_block.forward = types.MethodType(_forward_no_last_relu, last_block)

    def forward(self, x):
        pass

class SSP_MatchingNet(nn.Module):
    def __init__(self, pretrained=True, refine=False):
        super(SSP_MatchingNet, self).__init__()
        backbone = ResNet50_LoEC(pretrained=pretrained)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.refine = refine

        # Đóng băng tham số của layer0 và layer1 (theo LoEC.py dòng 27-40 & Bảng 1 bài báo)
        for p in self.layer0.parameters():
            p.requires_grad = False
        for p in self.layer1.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super(SSP_MatchingNet, self).train(mode)
        # Giữ layer0, layer1 và các tầng BatchNorm ở chế độ eval() theo chuẩn FSS/SSP
        self.layer0.eval()
        self.layer1.eval()
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    # -------------------------------------------------------------
    # LEM: Low-level Enhancement Module (LoEC.py dòng 255)
    # -------------------------------------------------------------
    def feat_RCfourier_bias(self, feat, b, h, w, in_channels, out_channels, kernel_size=3):
        random_convolution = nn.Conv2d(
            b * in_channels, b * out_channels, kernel_size, 
            padding=kernel_size // 2, groups=b, bias=True
        ).to(feat.device)
        torch.nn.init.kaiming_normal_(random_convolution.weight, mode="fan_in", nonlinearity="relu")

        feat_reshaped = feat.view(1, b * in_channels, h, w)
        feat_RC = random_convolution(feat_reshaped)
        feat_RC = feat_RC.view(b, out_channels, h, w)

        try:
            autocast_off = torch.amp.autocast(feat.device.type, enabled=False)
        except (AttributeError, TypeError):
            autocast_off = torch.cuda.amp.autocast(enabled=False)

        with autocast_off:
            feat_fp32 = feat.float()
            feat_RC_fp32 = feat_RC.float()
            fft_src = torch.fft.rfft2(feat_fp32, norm="backward")
            fft_trg = torch.fft.rfft2(feat_RC_fp32, norm="backward")

            amp_trg = torch.abs(fft_trg)
            pha_src = torch.angle(fft_src)
            real_part = amp_trg * torch.cos(pha_src)
            imag_part = amp_trg * torch.sin(pha_src)
            fft_mix = torch.complex(real_part, imag_part)

            out_lem = torch.fft.irfft2(fft_mix, s=(h, w), norm="backward")
            return out_lem.to(feat.dtype)

    def masked_average_pooling(self, feature, mask):
        mask_interp = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode="bilinear", align_corners=True)
        masked_feat = feature * mask_interp
        return masked_feat.sum(dim=(-2, -1)) / (mask_interp.sum(dim=(-2, -1)) + 1e-5)

    def similarity_func(self, feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)
        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out

    # -------------------------------------------------------------
    # LCM: Hỗ trợ cả Cosine Similarity (Bài báo) & Euclidean (LoEC.py)
    # -------------------------------------------------------------
    def mask_correction(self, out, q_ori, patch_size=16, K=3, method='cosine'):
        """
        LCM Module hỗ trợ 2 phương pháp tính độ tương đồng cấp thấp:
        - method='cosine': Chuẩn bài báo (arXiv:2503.21150v1 Eq. 10 & 11)
        - method='euclidean': Chuẩn mã nguồn LoEC.py tác giả
        """
        if q_ori.dim() == 4 and q_ori.shape[1] == 3:
            q_ori = q_ori.permute(0, 2, 3, 1)

        b, h, w = out.shape[0], out.shape[2], out.shape[3]
        out_confidence = out[:, 1, :, :] - out[:, 0, :, :]  # Eq. (7): C(i, j) = S(i, j, 1) - S(i, j, 0)

        num_patches_x = h // patch_size
        num_patches_y = w // patch_size
        patch_confidence = out_confidence.view(b, num_patches_x, patch_size, num_patches_y, patch_size).permute(0, 1, 3, 2, 4)
        patch_confidence_means = patch_confidence.mean(dim=(3, 4))  # Eq. (8): Patch pooling C_p

        total_patches = num_patches_x * num_patches_y
        k_val = min(K, total_patches)
        # Eq. (9): Chọn Top-K patch tin cậy nhất
        _, top_indices = torch.topk(patch_confidence_means.view(b, -1), k=k_val, dim=1, largest=True, sorted=True)

        reshaped_image = q_ori.reshape(b, num_patches_x, patch_size, num_patches_y, patch_size, 3).permute(0, 1, 3, 2, 4, 5)
        patch_means = reshaped_image.to(torch.float32).mean(dim=(3, 4))
        patch_means_reshaped = patch_means.view(b, -1, 3)  # [B, N, 3]

        if method == 'cosine':
            # ==============================================================
            # PHƯƠNG PHÁP 1: COSINE SIMILARITY (CHUẨN BÀI BÁO Eq. 10)
            # ==============================================================
            # Cosine similarity trực tiếp giữa vector màu cấp thấp của Top-K anchor và các patch
            patch_norm = F.normalize(patch_means_reshaped + 1e-7, p=2, dim=-1)  # [B, N, 3]
            anchor_norm = torch.gather(patch_norm, 1, top_indices.unsqueeze(-1).expand(-1, -1, 3))  # [B, K, 3]
            cosine_sim = torch.bmm(anchor_norm, patch_norm.transpose(1, 2))  # [B, K, N]
            sim_map = torch.clamp(cosine_sim, 0.0, 1.0)
            top_k_similarity_map = sim_map.view(b, k_val, num_patches_x, num_patches_y)
        else:
            # ==============================================================
            # PHƯƠNG PHÁP 2: EUCLIDEAN DISTANCE (CHUẨN MÃ NGUỒN LoEC.py)
            # ==============================================================
            diff = patch_means_reshaped.unsqueeze(1) - patch_means_reshaped.unsqueeze(2)
            euclidean_squared = torch.sum(diff ** 2, dim=-1)
            min_dist, _ = euclidean_squared.min(dim=2, keepdim=True)
            max_dist, _ = euclidean_squared.max(dim=2, keepdim=True)
            similarity = 1.0 - (euclidean_squared - min_dist) / (max_dist - min_dist + 1e-8)
            batch_indices = torch.arange(b, device=out.device).unsqueeze(1).expand_as(top_indices)
            top_k_similarity_map = similarity[batch_indices, top_indices, :].view(b, k_val, num_patches_x, num_patches_y)

        return top_indices, top_k_similarity_map

    # -------------------------------------------------------------
    # Self-Support Prototype Function (Khớp chuẩn LoEC.py dòng 178-240)
    # -------------------------------------------------------------
    def SSP_func(self, feature_q, out):
        bs = feature_q.shape[0]
        pred_1 = out.softmax(1).view(bs, 2, -1)
        pred_fg = pred_1[:, 1]
        pred_bg = pred_1[:, 0]

        fg_ls, bg_ls = [], []
        fg_local_ls, bg_local_ls = [], []

        for epi in range(bs):
            fg_thres = 0.7
            bg_thres = 0.6
            cur_feat = feature_q[epi].view(feature_q.shape[1], -1)
            f_h, f_w = feature_q[epi].shape[-2:]

            if (pred_fg[epi] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pred_fg[epi] > fg_thres)]
            else:
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 12).indices]

            if (pred_bg[epi] > bg_thres).sum() > 0:
                bg_feat = cur_feat[:, (pred_bg[epi] > bg_thres)]
            else:
                bg_feat = cur_feat[:, torch.topk(pred_bg[epi], 12).indices]

            fg_proto = fg_feat.mean(-1)
            bg_proto = bg_feat.mean(-1)
            fg_ls.append(fg_proto.unsqueeze(0))
            bg_ls.append(bg_proto.unsqueeze(0))

            fg_feat_norm = fg_feat / (torch.norm(fg_feat, 2, 0, True) + 1e-8)
            bg_feat_norm = bg_feat / (torch.norm(bg_feat, 2, 0, True) + 1e-8)
            cur_feat_norm = cur_feat / (torch.norm(cur_feat, 2, 0, True) + 1e-8)

            cur_feat_norm_t = cur_feat_norm.t()
            fg_sim = (torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0).softmax(-1)
            bg_sim = (torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0).softmax(-1)

            fg_proto_local = torch.matmul(fg_sim, fg_feat.t()).t().view(-1, f_h, f_w).unsqueeze(0)
            bg_proto_local = torch.matmul(bg_sim, bg_feat.t()).t().view(-1, f_h, f_w).unsqueeze(0)

            fg_local_ls.append(fg_proto_local)
            bg_local_ls.append(bg_proto_local)

        # Định dạng tensor prototype chuẩn 4D [B, 1024, 1, 1] để broadcast chính xác
        new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_fg_local = torch.cat(fg_local_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg_local = torch.cat(bg_local_ls, 0)

        return new_fg, new_bg, new_fg_local, new_bg_local

    def forward(self, img_s_list, mask_s_list, img_q, mask_q=None, 
                use_mask_correction=False, mask_k=0.6, mask_b=0.7, q_ori=None, use_lem=True,
                lcm_method='cosine', lcm_k=3, lcm_update_mode='additive'):
        b, h, w = img_q.shape[0], img_q.shape[2], img_q.shape[3]

        # 1. Trích xuất đặc trưng Support Images
        feature_s_list = []
        for k in range(len(img_s_list)):
            with torch.no_grad():
                s_0 = self.layer0(img_s_list[k])
                if self.training and use_lem:
                    s_0 = self.feat_RCfourier_bias(s_0, b, s_0.shape[2], s_0.shape[3], s_0.shape[1], s_0.shape[1], 3)
                s_0 = self.layer1(s_0)
            s_0 = self.layer2(s_0)
            s_0 = self.layer3(s_0)
            feature_s_list.append(s_0)

        # 2. Trích xuất đặc trưng Query Image
        with torch.no_grad():
            q_0 = self.layer0(img_q)
            q_0 = self.layer1(q_0)
        q_0 = self.layer2(q_0)
        feature_q = self.layer3(q_0)

        # 3. Trích xuất Prototype từ Support & tính Support Self-Similarity khi train
        feature_fg_list, feature_bg_list = [], []
        supp_out_ls = []
        for k in range(len(img_s_list)):
            feature_fg = self.masked_average_pooling(feature_s_list[k], (mask_s_list[k] == 1).float())[None, :]
            feature_bg = self.masked_average_pooling(feature_s_list[k], (mask_s_list[k] == 0).float())[None, :]
            feature_fg_list.append(feature_fg)
            feature_bg_list.append(feature_bg)

            if self.training:
                supp_sim_fg = F.cosine_similarity(feature_s_list[k], feature_fg.squeeze(0)[..., None, None], dim=1)
                supp_sim_bg = F.cosine_similarity(feature_s_list[k], feature_bg.squeeze(0)[..., None, None], dim=1)
                supp_out = torch.cat((supp_sim_bg[:, None, ...], supp_sim_fg[:, None, ...]), dim=1) * 10.0
                supp_out = F.interpolate(supp_out, size=(h, w), mode="bilinear", align_corners=True)
                supp_out_ls.append(supp_out)

        # Prototype Support chuẩn 4D [B, 1024, 1, 1]
        FP = torch.mean(torch.cat(feature_fg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        BP = torch.mean(torch.cat(feature_bg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)

        # 4. Đo độ tương đồng giữa query feature và prototype
        out_0 = self.similarity_func(feature_q, FP, BP)

        SSFP_1, SSBP_1, ASFP_1, ASBP_1 = self.SSP_func(feature_q, out_0)
        FP_1 = FP * 0.5 + SSFP_1 * 0.5
        BP_1 = SSBP_1 * 0.3 + ASBP_1 * 0.7  # [B, 1024, 1, 1] + [B, 1024, H, W] -> broadcast chuẩn xác!
        out_1 = self.similarity_func(feature_q, FP_1, BP_1)

        if self.refine:
            SSFP_2, SSBP_2, ASFP_2, ASBP_2 = self.SSP_func(feature_q, out_1)
            FP_2 = FP * 0.5 + SSFP_2 * 0.5
            BP_2 = SSBP_2 * 0.3 + ASBP_2 * 0.7
            FP_2 = FP * 0.5 + FP_1 * 0.2 + FP_2 * 0.3
            BP_2 = BP * 0.5 + BP_1 * 0.2 + BP_2 * 0.3
            out_2 = self.similarity_func(feature_q, FP_2, BP_2)
            out_2 = out_2 * 0.7 + out_1 * 0.3
            out_2 = F.interpolate(out_2, size=(h, w), mode="bilinear", align_corners=True)
            out_1 = F.interpolate(out_1, size=(h, w), mode="bilinear", align_corners=True)
            out_ls = [out_2, out_1]
        else:
            out_1 = F.interpolate(out_1, size=(h, w), mode="bilinear", align_corners=True)
            out_ls = [out_1]

        if self.training and (mask_q is not None):
            fg_q = self.masked_average_pooling(feature_q, (mask_q == 1).float())[None, :].squeeze(0)
            bg_q = self.masked_average_pooling(feature_q, (mask_q == 0).float())[None, :].squeeze(0)
            self_sim_fg = F.cosine_similarity(feature_q, fg_q[..., None, None], dim=1)
            self_sim_bg = F.cosine_similarity(feature_q, bg_q[..., None, None], dim=1)
            self_out = torch.cat((self_sim_bg[:, None, ...], self_sim_fg[:, None, ...]), dim=1) * 10.0
            self_out = F.interpolate(self_out, size=(h, w), mode="bilinear", align_corners=True)
            supp_out = torch.cat(supp_out_ls, 0)
            out_ls.append(self_out)
            out_ls.append(supp_out)

        # 5. LCM khi kiểm thử trên miền đích
        if use_mask_correction and (q_ori is not None):
            top_k_loc, top_k_sim = self.mask_correction(
                out_ls[0], q_ori, patch_size=16, K=lcm_k, method=lcm_method
            )
            reshaped_sim = F.interpolate(top_k_sim, (h, w), mode='nearest')
            if lcm_update_mode == 'additive':
                # Eq. (11): Cập nhật dạng cộng tuyến tính của bài báo
                delta = torch.sum(mask_k * (reshaped_sim - mask_b), dim=1, keepdim=True)
                out_ls[0][:, 1:2, :, :] = out_ls[0][:, 1:2, :, :] + delta
            else:
                # Dạng nhân tỷ lệ của LoEC.py
                out_ls[0][:, 1, :, :] = out_ls[0][:, 1, :, :] + out_ls[0][:, 1, :, :] * (reshaped_sim[:, 0, :, :] - mask_b) * mask_k

        if self.training and (mask_q is not None):
            return out_ls
        return out_ls[0]

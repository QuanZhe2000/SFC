import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
import numpy as np
from ..builder import DISTILL_LOSSES


class FactorTransfer(nn.Module):
    """Paraphrasing Complex Network: Network Compression via Factor Transfer, NeurIPS 2018"""

    def __init__(self, p1=2, p2=1, name="factor_transfer"):
        super(FactorTransfer, self).__init__()
        self.p1 = p1
        self.p2 = p2

    def forward(self, f_s, f_t):
        return self.factor_loss(f_s, f_t)

    def factor_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[1], f_t.shape[1]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        if self.p2 == 1:
            return (self.factor(f_s) - self.factor(f_t)).abs().mean()
        else:
            return (self.factor(f_s) - self.factor(f_t)).pow(self.p2).mean()

    def factor(self, f):
        return F.normalize(f.pow(self.p1).mean(1).view(f.size(0), -1))


def feature_norm(feat):
    assert len(feat.shape) == 4
    N, C, H, W = feat.shape
    feat = feat.permute(1, 0, 2, 3).reshape([C, -1])
    # feat = feat.transpose([1, 0, 2, 3]).reshape([C, -1])
    mean = feat.mean(axis=-1, keepdim=True)
    std = feat.std(axis=-1, keepdim=True)
    feat = (feat - mean) / (std + 1e-6)
    return feat.reshape([C, N, H, W]).permute(1, 0, 2, 3)
    # return feat.reshape([C, N, H, W]).transpose([1, 0, 2, 3])


@DISTILL_LOSSES.register_module()
class fg_bgUpdateLoss(nn.Module):
    """PyTorch version of `Feature Distillation for General Detectors`

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        temp (float, optional): Temperature coefficient. Defaults to 0.5.
        name (str): the loss name of the layer
        alpha_fgd (float, optional): Weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): Weight of bg_loss. Defaults to 0.0005
        gamma_fgd (float, optional): Weight of mask_loss. Defaults to 0.0005
        lambda_fgd (float, optional): Weight of relation_loss. Defaults to 0.000005
    """

    def __init__(
        self,
        student_channels,
        teacher_channels,
        name,
        temp=0.5,
        alpha_fgd=0.001,
        beta_fgd=0.0005,
        gamma_fgd=0.0005,
        lambda_fgd=0.000005,
        fg_bg_use=1,
    ):
        super(fg_bgUpdateLoss, self).__init__()
        self.fg_bg_use = fg_bg_use
        self.fg_list = [1, 2, 3, 4, 5, 6, 7, 8]
        self.bg_list = [2, 4, 5, 7]
        self.cwd_list = [3, 5, 6, 7]
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd
        self.tau = 1.0
        self.loss_weight = 1.0
        self.eps = 1e-8

        self.alpha_mgd = 0.00002
        self.lambda_mgd = 0.65

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(
                student_channels, teacher_channels, kernel_size=1, stride=1, padding=0
            )
        else:
            self.align = None

        # self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        # self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.ft = FactorTransfer()

        self.conv_mask_s=nn.Sequential(
            nn.Conv2d(teacher_channels,teacher_channels,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(teacher_channels,1,kernel_size=1),
        )
        self.conv_mask_t=nn.Sequential(
            nn.Conv2d(teacher_channels,teacher_channels,kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(teacher_channels,1,kernel_size=1),
        )
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels // 2, kernel_size=1),
            # nn.LayerNorm([teacher_channels // 2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels // 2, teacher_channels, kernel_size=1),
        )
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels // 2, kernel_size=1),
            # nn.LayerNorm([teacher_channels // 2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels // 2, teacher_channels, kernel_size=1),
        )

        self.reset_parameters()

    def Gaussin(self, x, y, x_center, y_center):
        d = (-1 / 2.0) * (
            (x - x_center) / (x_center + self.eps)
            + (y - y_center) / (y_center + self.eps)
        ) ** 2
        return (torch.exp(d)).cuda()

    def Gaussin_matrix(self, x, y, x_center, y_center):
        d = (-1 / 2.0) * (
            (x - x_center) / (x_center + self.eps)
            + (y - y_center) / (y_center + self.eps)
        ) ** 2
        return (torch.exp(d)).cuda()

    def Gaussin_matrix2(self, x, y, x_center, y_center):
        d = (-1 / 2.0) * (
            (x - x_center) ** 2 / (x_center + self.eps) ** 2
            + (y - y_center) ** 2 / (y_center + self.eps) ** 2
        )
        return (torch.exp(d)).cuda()

    # def use_gaussin():
    #     for j in range(len(gt_bboxes[i])):
    #         a = torch.linspace(hmin[i][j], hmax[i][j] , hmax[i][j]  - hmin[i][j])
    #         b = torch.linspace(wmin[i][j], wmax[i][j] , wmax[i][j]  - wmin[i][j])
    #         a = a.reshape(-1, 1).cuda()
    #         b = b.reshape(1, -1).cuda()
    #         data = self.Gaussin(a, b, x_center[j], y_center[j])
    #         data = area[0][j] * data
    #         Mask_fg[i][hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]] = \
    #             torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]], data)
    """
        1.Gaussin+fg,bg CWD
        2.高斯 fg做CWD
        3.高斯 bg做CWD
        4.无fg+bg高斯，CWD
        5.纯FGD，无fg，bg，CWD
        6.fg
        7.bg
    """

    def forward(self, preds_S, preds_T, gt_bboxes, scale_y, scale_x, img_metas):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
            gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
            img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        """

        assert (
            preds_S.shape[-2:] == preds_T.shape[-2:]
        ), "the output dim of teacher and student differ"

        if self.align is not None:
            preds_S = self.align(preds_S)

        N, C, H, W = preds_S.shape

        S_attention_t, C_attention_t = self.get_attention(preds_T, self.temp)
        S_attention_s, C_attention_s = self.get_attention(preds_S, self.temp)

        Mask_fg = torch.zeros_like(S_attention_t)
        Mask_bg = torch.ones_like(S_attention_t)

        wmin, wmax, hmin, hmax = [], [], [], []

        for i in range(N):
            new_boxxes = torch.ones_like(gt_bboxes[i])
            new_boxxes[:, 0] = (
                gt_bboxes[i][:, 0] / img_metas[i]["img_shape"][1] * W / scale_x
            )
            new_boxxes[:, 2] = (
                gt_bboxes[i][:, 2] / img_metas[i]["img_shape"][1] * W / scale_x
            )
            new_boxxes[:, 1] = (
                gt_bboxes[i][:, 1] / img_metas[i]["img_shape"][0] * H / scale_y
            )
            new_boxxes[:, 3] = (
                gt_bboxes[i][:, 3] / img_metas[i]["img_shape"][0] * H / scale_y
            )

            # bbox
            wmin.append(torch.floor(new_boxxes[:, 0]).int())
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            x_center = (new_boxxes[:, 0] + new_boxxes[:, 2]) / 2
            y_center = (new_boxxes[:, 1] + new_boxxes[:, 3]) / 2

            """
                FGD将Scale和Mask直接相乘得到了area，避免再次计算
                我需要将其分开，进行高斯运算后再乘以尺度不变性S

            """
            area = (
                1.0
                / (hmax[i].view(1, -1) + 1 - hmin[i].view(1, -1))
                / (wmax[i].view(1, -1) + 1 - wmin[i].view(1, -1))
            )

            """
                对于bbox外边一圈作高斯
            """
            H = torch.tensor(H).to(torch.int32).cuda()
            W = torch.tensor(W).to(torch.int32).cuda()
            hmax[i] = torch.where(hmax[i] < H, hmax[i], H)
            wmax[i] = torch.where(wmax[i] < W, wmax[i], W)
            for j in range(len(gt_bboxes[i])):
                d_h = (hmax[i][j] - hmin[i][j]) // 2
                start_h = max(0, hmin[i][j] - d_h)
                end_h = min(hmax[i][j] + d_h, H)
                a = torch.linspace(start_h, end_h, end_h - start_h)
                d_w = (wmax[i][j] - wmin[i][j]) // 2
                start_w = max(0, wmin[i][j] - d_w)
                end_w = min(wmax[i][j] + d_w, W)
                b = torch.linspace(start_w, end_w, end_w - start_w)
                a = a.reshape(-1, 1).cuda()
                b = b.reshape(1, -1).cuda()
                data = self.Gaussin_matrix2(a, b, x_center[j], y_center[j])
                data = area[0][j] * data
                Mask_fg[i][start_h:end_h, start_w:end_w] = torch.maximum(
                    Mask_fg[i][start_h:end_h, start_w:end_w], data
                )
            # bbox部分设置为1
            for j in range(len(gt_bboxes[i])):
                Mask_fg[i][hmin[i][j] : hmax[i][j] + 1, wmin[i][j] : wmax[i][j] + 1] = (
                    torch.maximum(
                        Mask_fg[i][
                            hmin[i][j] : hmax[i][j] + 1, wmin[i][j] : wmax[i][j] + 1
                        ],
                        area[0][j],
                    )
                )

            Mask_bg[i] = torch.where(Mask_fg[i] > 0, 0, 1)
            if torch.sum(Mask_bg[i]):
                Mask_bg[i] /= torch.sum(Mask_bg[i])
        # 使用CWD和高斯改造前景背景部分
        # loss = fg_loss + self.beta_fgd * bg_loss
        # 1:Gaussion使用，CWD使用

        fg_loss, bg_loss, ft_loss = self.get_fea_loss(
            preds_S,
            preds_T,
            Mask_fg,
            Mask_bg,
            C_attention_s,
            C_attention_t,
            S_attention_s,
            S_attention_t,
        )
        mask_loss = self.get_mask_loss(
            C_attention_s, C_attention_t, S_attention_s, S_attention_t
        )
        rela_loss = self.get_rela_loss(preds_S, preds_T)
        loss=fg_loss+bg_loss+ self.gamma_fgd * mask_loss + self.gamma_fgd * rela_loss
        # loss = fg_loss + bg_loss
        # if self.fg_bg_use == 1:
        #     loss += self.gamma_fgd * mask_loss
        # elif self.fg_bg_use == 2:
        #     loss += self.gamma_fgd * mask_loss + self.gamma_fgd * rela_loss
        return loss

    def get_attention(self, preds, temp):
        """preds: Bs*C*W*H"""
        N, C, H, W = preds.shape

        value = torch.abs(preds)
        # Bs*W*H
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = (H * W * F.softmax((fea_map / temp).view(N, -1), dim=1)).view(
            N, H, W
        )

        # Bs*C
        channel_map = value.mean(axis=2, keepdim=False).mean(axis=2, keepdim=False)
        C_attention = C * F.softmax(channel_map / temp, dim=1)

        return S_attention, C_attention

    def distill_softmax(self, x, tau):
        _, _, w, h = x.shape
        x = x.contiguous().view(-1, w * h)
        x /= tau
        return F.softmax(x, dim=1)

    def get_mask_loss(self, C_s, C_t, S_s, S_t):

        mask_loss = torch.sum(torch.abs((C_s - C_t))) / len(C_s) + torch.sum(
            torch.abs((S_s - S_t))
        ) / len(S_s)
        # mask_loss_C=self.ft(C_s, C_t)
        # mask_loss_S=self.ft(S_s, S_t)

        return mask_loss

    def spatial_pool(self, x, in_type):
        batch, channel, width, height = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        if in_type == 0:
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = F.softmax(context_mask, dim=2)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def get_rela_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction="sum")

        context_s = self.spatial_pool(preds_S, 0)
        context_t = self.spatial_pool(preds_T, 1)

        out_s = preds_S
        out_t = preds_T

        channel_add_s = self.channel_add_conv_s(context_s)
        out_s = out_s + channel_add_s

        channel_add_t = self.channel_add_conv_t(context_t)
        out_t = out_t + channel_add_t

        rela_loss = loss_mse(out_s, out_t) / len(out_s)

        return rela_loss
    def get_rela_loss2(self, preds_S, preds_T):
        out_s = self.conv_mask_s(preds_S)
        out_t = self.conv_mask_t(preds_T)
        
        loss_mse = nn.MSELoss(reduction="sum")

        rela_loss = loss_mse(out_s, out_t) / len(out_s)

        return rela_loss

    def get_fea_loss(self, preds_S, preds_T, Mask_fg, Mask_bg, C_s, C_t, S_s, S_t):
        loss_mse = nn.MSELoss(reduction="sum")

        Mask_fg = Mask_fg.unsqueeze(dim=1)
        Mask_bg = Mask_bg.unsqueeze(dim=1)

        C_t = C_t.unsqueeze(dim=-1)
        C_t = C_t.unsqueeze(dim=-1)

        S_t = S_t.unsqueeze(dim=1)

        fea_t = torch.mul(preds_T, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))

        fea_s = torch.mul(preds_S, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg))
        ft_loss = 0
        N, C, H, W = fg_fea_s.shape
        # # 进行归一化，做KL
        fg_fea_t = feature_norm(fg_fea_t)
        fg_fea_s = feature_norm(fg_fea_s)
        fg_fea_t = self.distill_softmax(fg_fea_t, tau=1.0)
        fg_fea_s = self.distill_softmax(fg_fea_s, tau=1.0)
        eps = 1e-5
        loss_fg = torch.sum(
            -fg_fea_t * torch.log(eps + fg_fea_s) + fg_fea_t * torch.log(eps + fg_fea_t)
        )
        fg_loss = loss_fg / (C * N)
        N, C, H, W = bg_fea_s.shape
        # # 进行归一化，做KL
        bg_fea_t = feature_norm(bg_fea_t)
        bg_fea_s = feature_norm(bg_fea_s)
        bg_fea_t = self.distill_softmax(bg_fea_t, tau=1.0)
        bg_fea_s = self.distill_softmax(bg_fea_s, tau=1.0)
        eps = 1e-5
        loss_bg = torch.sum(
            -bg_fea_t * torch.log(eps + bg_fea_s) + bg_fea_t * torch.log(eps + bg_fea_t)
        )
        bg_loss = loss_bg / (C * N)

        return fg_loss, bg_loss, ft_loss

    def last_zero_init(self, m):
        if isinstance(m, nn.Sequential):
            constant_init(m[-1], val=0)
        else:
            constant_init(m, val=0)

    def reset_parameters(self):
        kaiming_init(self.conv_mask_s, mode="fan_in")
        kaiming_init(self.conv_mask_t, mode="fan_in")
        # kaiming_init(self.generation, mode='fan_in')

        self.conv_mask_s.inited = True
        self.conv_mask_t.inited = True
        # self.generation.inited = True
        self.last_zero_init(self.channel_add_conv_s)
        self.last_zero_init(self.channel_add_conv_t)

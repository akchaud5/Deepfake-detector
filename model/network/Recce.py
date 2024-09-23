import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models import xception
from model.common import SeparableConv2d, Block, GuidedAttention, GraphReasoning

class FeedForward2D(nn.Module):
    def __init__(self, input_dim=728, output_dim=728):
        super(FeedForward2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)

class GlobalFilter(nn.Module):
    def __init__(self, dim=728, h=20, w=11, fp32fft=True):
        super().__init__()
        self.fp32fft = fp32fft
        self.complex_weight = nn.Parameter(
            torch.randn(h, w // 2 + 1, dim, 2, dtype=torch.float32) * 0.02
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()

        if self.fp32fft:
            dtype = x.dtype
            x = x.to(torch.float32)

        x_fft = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")

        weight_real = torch.randn(h, w // 2 + 1, c, dtype=torch.float32, device=x_fft.device) * 0.02
        weight_imag = torch.randn(h, w // 2 + 1, c, dtype=torch.float32, device=x_fft.device) * 0.02
        weight = torch.complex(weight_real, weight_imag).unsqueeze(0)
        weight = weight.expand(b, -1, -1, -1)

        x_fft = x_fft * weight

        x_ifft = torch.fft.irfft2(x_fft, s=(h, w), dim=(1, 2), norm="ortho")

        if self.fp32fft:
            x_ifft = x_ifft.to(dtype)

        return x_ifft.permute(0, 3, 1, 2).contiguous()

class FreqBlock(nn.Module):
    def __init__(self, dim, fp32fft=True):
        super().__init__()
        self.filter = GlobalFilter(dim, fp32fft=fp32fft)
        self.feed_forward = FeedForward2D(dim, dim)

    def forward(self, x):
        x = self.filter(x)
        x = x + self.feed_forward(x)
        return x

class CMA_Block(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super(CMA_Block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )
        self.conv3 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )

        self.scale = hidden_channel ** -0.5

        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, rgb, freq):
        _, _, h, w = rgb.size()

        q = self.conv1(rgb)
        k = self.conv2(freq)
        v = self.conv3(freq)

        q = q.view(q.size(0), q.size(1), q.size(2) * q.size(3)).transpose(-2, -1)
        k = k.view(k.size(0), k.size(1), k.size(2) * k.size(3))

        attn = torch.matmul(q, k) * self.scale
        m = attn.softmax(dim=-1)

        v = v.view(v.size(0), v.size(1), v.size(2) * v.size(3)).transpose(-2, -1)
        z = torch.matmul(m, v)
        z = z.view(z.size(0), h, w, -1)
        z = z.permute(0, 3, 1, 2).contiguous()

        return rgb + self.conv4(z)

encoder_params = {
    "xception": {
        "features": 2048,
        "init_op": partial(xception, pretrained=True)
    }
}

class Recce(nn.Module):
    """End-to-End Reconstruction-Classification Learning for Face Forgery Detection"""

    def __init__(self, num_classes=2, drop_rate=0.2):
        super(Recce, self).__init__()
        self.loss_inputs = dict()
        self.encoder = encoder_params["xception"]["init_op"]()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(encoder_params["xception"]["features"], num_classes)

        self.attention = GuidedAttention(depth=728, drop_rate=drop_rate)
        # self.reasoning = GraphReasoning(728, 256, 256, 256, 128, 256, [2, 4], drop_rate)

        self.decoder1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(728, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = Block(256, 256, 3, 1)
        self.decoder3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder4 = Block(128, 128, 3, 1)
        self.decoder5 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder6 = nn.Sequential(
            nn.Conv2d(64, 3, 1, 1, bias=False),
            nn.Tanh()
        )

        self.filter = FreqBlock(dim=728, fp32fft=True)
        self.cma = CMA_Block(in_channel=728, hidden_channel=728, out_channel=728)  # Updated channels

    def norm_n_corr(self, x):
        norm_embed = F.normalize(self.global_pool(x), p=2, dim=1)
        corr = (torch.matmul(norm_embed.squeeze(), norm_embed.squeeze().T) + 1.) / 2.
        return norm_embed, corr

    @staticmethod
    def add_white_noise(tensor, mean=0., std=1e-6):
        rand = torch.rand([tensor.shape[0], 1, 1, 1])
        rand = torch.where(rand > 0.5, 1., 0.).to(tensor.device)
        white_noise = torch.normal(mean, std, size=tensor.shape, device=tensor.device)
        noise_t = tensor + white_noise * rand
        return torch.clip(noise_t, -1., 1.)

    def forward(self, x):
        self.loss_inputs = dict(recons=[], contra=[])

        noise_x = self.add_white_noise(x) if self.training else x

        out = self.encoder.conv1(noise_x)
        out = self.encoder.bn1(out)
        out = self.encoder.act1(out)
        out = self.encoder.conv2(out)
        out = self.encoder.bn2(out)
        out = self.encoder.act2(out)
        out = self.encoder.block1(out)
        out = self.encoder.block2(out)
        out = self.encoder.block3(out)
        embedding = self.encoder.block4(out)

        norm_embed, corr = self.norm_n_corr(embedding)
        self.loss_inputs['contra'].append(corr)

        out = self.dropout(embedding)
        out = self.decoder1(out)
        out_d2 = self.decoder2(out)

        norm_embed, corr = self.norm_n_corr(out_d2)
        self.loss_inputs['contra'].append(corr)

        out = self.decoder3(out_d2)
        out_d4 = self.decoder4(out)

        norm_embed, corr = self.norm_n_corr(out_d4)
        self.loss_inputs['contra'].append(corr)

        out = self.decoder5(out_d4)
        pred = self.decoder6(out)

        recons_x = F.interpolate(pred, size=x.shape[-2:], mode='bilinear', align_corners=True)
        self.loss_inputs['recons'].append(recons_x)

        embedding = self.encoder.block5(embedding)
        embedding = self.encoder.block6(embedding)
        embedding = self.encoder.block7(embedding)

        freq = self.filter(embedding)

        fusion = self.cma(embedding, freq) + embedding

        embedding = self.encoder.block8(fusion)
        img_att = self.attention(x, recons_x, embedding)

        embedding = self.encoder.block9(img_att)
        embedding = self.encoder.block10(embedding)
        embedding = self.encoder.block11(embedding)
        embedding = self.encoder.block12(embedding)

        embedding = self.encoder.conv3(embedding)
        embedding = self.encoder.bn3(embedding)
        embedding = self.encoder.act3(embedding)
        embedding = self.encoder.conv4(embedding)
        embedding = self.encoder.bn4(embedding)
        embedding = self.encoder.act4(embedding)

        embedding = self.global_pool(embedding).squeeze()

        out = self.dropout(embedding)
        return self.fc(out)

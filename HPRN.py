import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from skimage import segmentation


class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(Conv3x3, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class SSRM(nn.Module):  # Semantic-embed Spatial Relation Module
    def __init__(self, channels=200, window_size=8*8, n_scales=4, reduction=4):
        super(SSRM, self).__init__()
        self.window_size = window_size
        self.n_scales = n_scales
        self.reduction = reduction
        self.conv_x = Conv3x3(channels, channels // reduction, 3, 1)
        self.conv_y = Conv3x3(channels, channels, 1, 1)
        self.conv_out = nn.Conv2d(channels*n_scales, channels, 1)

    def semantic_prior_embed(self, semantic_label):
        N, _, H, W = semantic_label.shape
        device = semantic_label.device
        semantic_label = torch.reshape(semantic_label, (N, -1, H*W))  # [N,n_scales,H*W]
        # add offsets to avoid semantic label overlapping between different scales
        offsets = torch.arange(self.n_scales, device=device)
        offsets = torch.reshape(offsets * 100, (1, -1, 1))
        semantic_label = torch.reshape(semantic_label + offsets, (N, -1))  # [N,n_scales*H*W]
        return semantic_label

    def batched_index_select(self, values, indices):
        last_dim = values.shape[-1]
        return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

    def add_adjacent_clusters(self, x):
        x_extra_back = torch.cat([x[:, :, -1:, ...], x[:, :, :-1, ...]], dim=2)
        x_extra_forward = torch.cat([x[:, :, 1:, ...], x[:, :, :1, ...]], dim=2)
        return torch.cat([x, x_extra_back, x_extra_forward], dim=3)

    def forward(self, input, semantic_label):  # N,C,H,W; N,n_scales,H,W
        N, _, H, W = input.shape
        x_embed = self.conv_x(input).view(N, -1, H * W).permute(0, 2, 1).contiguous()
        y_embed = self.conv_y(input).view(N, -1, H * W).permute(0, 2, 1).contiguous()
        L, C = x_embed.shape[-2:]
        # add offsets and reshape dim
        semantic_label = self.semantic_prior_embed(semantic_label)
        semantic_label = semantic_label.detach()
        # regroup elements with same semantic label by sorting
        _, indices = semantic_label.sort(dim=-1)  # [N,n_scales*H*W]
        _, undo_sort = indices.sort(dim=-1)  # undo_sort to recover original order
        mod_indices = (indices % L)  # now range from (0->H*W)
        x_embed_sorted = self.batched_index_select(x_embed, mod_indices)  # [N,n_scales*H*W,C]
        y_embed_sorted = self.batched_index_select(y_embed, mod_indices)  # [N,n_scales*H*W,C]
        # pad the embedding if it cannot be divided by window_size
        padding = self.window_size - L % self.window_size if L % self.window_size != 0 else 0
        x_relation_clusters = torch.reshape(x_embed_sorted, (N, self.n_scales, -1, C))  # [N, n_scales, H*W,C]
        y_relation_clusters = torch.reshape(y_embed_sorted, (N, self.n_scales, -1, C * self.reduction))
        if padding:
            pad_x = x_relation_clusters[:, :, -padding:, :].clone()
            pad_y = y_relation_clusters[:, :, -padding:, :].clone()
            x_relation_clusters = torch.cat([x_relation_clusters, pad_x], dim=2)
            y_relation_clusters = torch.cat([y_relation_clusters, pad_y], dim=2)
        x_relation_clusters = torch.reshape(x_relation_clusters, (
        N, self.n_scales, -1, self.window_size, C))  # [N, n_scales, num_windows, window_size, C]
        y_relation_clusters = torch.reshape(y_relation_clusters, (N, self.n_scales, -1, self.window_size, C * self.reduction))
        # normalize to control numerical stability
        x_match = F.normalize(x_relation_clusters, p=2, dim=-1, eps=5e-5)
        # allow relation to adjacent clusters
        x_match = self.add_adjacent_clusters(x_match)
        y_relation_clusters = self.add_adjacent_clusters(y_relation_clusters)
        # acquire relation score
        raw_weight = torch.einsum('bhkie,bhkje->bhkij', x_relation_clusters, x_match)  # [N, n_scales, num_windows, window_size, window_size*3]
        # softmax
        cluster_weight = torch.logsumexp(raw_weight, dim=-1, keepdim=True)
        weight = torch.exp(raw_weight - cluster_weight)  # (after softmax)
        # update out with relation score based on cluster dim
        out = torch.einsum('bukij,bukje->bukie', weight, y_relation_clusters)  # [N, n_scales, num_windows, window_size, C]
        out = torch.reshape(out, (N, self.n_scales, -1, C * self.reduction))
        # if padded, then remove extra elements
        if padding:
            out = out[:, :, :-padding, :].clone()
        # recover the original order
        out = torch.reshape(out, (N, -1, C * self.reduction))  # [N,n_scales*H*W,C]
        out = self.batched_index_select(out, undo_sort)  # [N,n_scales*H*W,C]
        # map multi-scale out to original dim
        out = torch.reshape(out, (N, self.n_scales, L, C * self.reduction))  # [N,n_scales,H*W,C]
        out = self.conv_out(out.permute(0, 1, 3, 2).contiguous().view(N, -1, H, W)) + input
        return out


class MSAModule(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=False, kdim=None, vdim=None, batch_first=False):
        super().__init__()
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, q, k, v, need_weights=False):
        if self.batch_first:
            q, k, v = [x.transpose(0, 1) for x in (q, k, v)]
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        tgt_len, bsz, _ = q.size()
        src_len = k.size(0)
        # q: bsz * num_heads, tgt_len, head_dim
        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # k, v: bsz * num_heads, src_len, head_dim
        k, v = [x.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1) for x in (k, v)]
        # 计算att
        att_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale  # bsz * num_heads, tgt_len, src_len
        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.bmm(att_weights, v)  # bsz * num_heads, tgt_len, head_dim
        if not self.batch_first:
            out = out.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        else:
            out = out.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim).transpose(0, 1)
        out = self.out_proj(out)
        if need_weights:
            # average attention weights over heads
            attn_output_weights = att_weights.view(bsz, self.num_heads, tgt_len, src_len)
            return out, attn_output_weights.mean(dim=1)
        else:
            return out


class TSRM(nn.Module):  # Tranformer-based Spectral Relation Module
    def __init__(self, channel, reduction=16, patch_num=2):
        super(TSRM, self).__init__()
        self.patch_num = patch_num
        self.avg_pool_embed = nn.AdaptiveAvgPool2d(patch_num)
        self.cross_transformer = MSAModule(embed_dim=patch_num*patch_num, num_heads=1, bias=False, batch_first=True)
        self.gate = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.PReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        batch, channel, _, _ = x.size()
        # Spectral Relation Interaction
        x_avg_embed = self.avg_pool_embed(x).view(batch, channel, -1)
        x_avg_embed = self.cross_transformer(x_avg_embed, x_avg_embed, x_avg_embed)
        cross_vector = x_avg_embed.mean(-1).view(batch, channel, 1, 1)
        att = self.gate(cross_vector)
        x = x * att
        return x


class MRB(nn.Module):  # Multi-Residual Block
    def __init__(self, in_dim, out_dim, patch_num=2):
        super(MRB, self).__init__()
        self.conv0 = Conv3x3(in_dim, out_dim, 3, 1)
        self.act0 = nn.PReLU()
        self.conv1 = Conv3x3(out_dim, out_dim, 3, 1)
        self.act1 = nn.PReLU()
        self.conv2 = Conv3x3(out_dim, out_dim, 3, 1)
        self.act2 = nn.PReLU()
        self.conv3 = Conv3x3(out_dim, out_dim, 3, 1)
        self.tsrm = TSRM(out_dim, reduction=16, patch_num=patch_num)
        self.act3 = nn.PReLU()

    def forward(self, x):
        res = x
        x = self.conv0(x)
        x = self.act0(x)
        x = self.conv1(x)
        x = self.act1(x)
        x += res
        x = self.conv2(x)
        x += res
        x = self.act2(x)
        x = self.conv3(x)
        x = self.tsrm(x)
        x += res
        x = self.act3(x)

        return x


class HPRN(nn.Module):
    def __init__(self, inplanes=3, outplanes=31, interplanes=200, n_DRBs=10, window_size=8*8, n_scales=4, patch_num=4):
        super(HPRN, self).__init__()
        self.n_DRBs = n_DRBs
        self.head_conv2D = Conv3x3(inplanes, interplanes, 3, 1)

        self.backbone = nn.ModuleList(
            [MRB(in_dim=interplanes, out_dim=interplanes, patch_num=patch_num) for _ in range(n_DRBs)])

        self.tail_conv2D = Conv3x3(interplanes, interplanes, 3, 1)
        self.output_conv2D = Conv3x3(interplanes, outplanes, 3, 1)
        self.ssrm = SSRM(channels=outplanes, window_size=window_size, n_scales=n_scales, reduction=1)

    def forward(self, x, semantic_label):
        out = self.head_conv2D(x)
        residual = out

        for _, block in enumerate(self.backbone):
            out = block(out)

        out = self.tail_conv2D(out)
        out += residual
        out = self.output_conv2D(out)
        out = self.ssrm(out, semantic_label)
        return out


if __name__ == "__main__":
    # 1 Convert cv_image to numpy
    rgb = np.random.randint(low=0, high=256, size=3*64*64, dtype='uint8')
    rgb = rgb.reshape(64, 64, 3)
    # 2 Acquire SLIC result
    semantic_label_list = []
    slic_scales = [8, 12, 16, 20]
    slic_input = np.uint8(rgb * 255.0)
    for s in slic_scales:
        semantic_label_list.append(segmentation.slic(slic_input, start_label=1, n_segments=s)[None, :])
    semantic_labels = np.concatenate(semantic_label_list, axis=0)
    # 3 Convert numpy to torch
    rgb = np.transpose(rgb, [2, 0, 1])
    rgb = np.float32(rgb) / 255.0
    rgb = torch.from_numpy(rgb[None, :])
    semantic_labels = torch.Tensor(semantic_labels[None, :])
    # 4 Input model
    model = HPRN()
    with torch.no_grad():
        output_tensor = model(rgb, semantic_labels)
    print(output_tensor.size())
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))

    from thop import profile
    from thop import clever_format
    macs, params = profile(model, inputs=(rgb, semantic_labels))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"macs: {macs}, params: {params}")



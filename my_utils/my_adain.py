import torch
import torch.nn.functional as F

def calc_mean_std(feat, eps=1e-6):
    """
    Calculate mean and standard deviation of feature maps.
    Args:
        feat (torch.Tensor): Input tensor of shape [B, C, H, W] or [C, H, W].
        eps (float): Small value to avoid division by zero.
    Returns:
        mean (torch.Tensor): Mean of shape [B, C, 1, 1] or [C, 1, 1].
        std (torch.Tensor): Standard deviation of shape [B, C, 1, 1] or [C, 1, 1].
    """
    if feat.dim() == 3:
        feat = feat.unsqueeze(0)
    size = feat.size()
    assert len(size) == 4, f"Expected 4D tensor, got shape {size}"
    batch, channels = size[0], size[1]
    
    feat = feat.view(batch, channels, -1)
    mean = feat.mean(dim=2, keepdim=True)
    var = feat.var(dim=2, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    
    mean = mean.view(batch, channels, 1, 1)
    std = std.view(batch, channels, 1, 1)
    
    return mean, std

def adain(content_feat, style_feat):
    """
    Adaptive Instance Normalization.
    Args:
        content_feat (torch.Tensor): Content features, shape [B, C, H, W] or [C, H, W].
        style_feat (torch.Tensor): Style features, shape [B, C, H, W] or [C, H, W].
    Returns:
        torch.Tensor: Normalized content features with style statistics, shape [B, C, H, W].
    """
    if content_feat.dim() == 3:
        content_feat = content_feat.unsqueeze(0)
    if style_feat.dim() == 3:
        style_feat = style_feat.unsqueeze(0)
    
    assert content_feat.size()[:2] == style_feat.size()[:2], \
        f"Shape mismatch: content {content_feat.size()} vs style {style_feat.size()}"
    
    content_mean, content_std = calc_mean_std(content_feat)
    style_mean, style_std = calc_mean_std(style_feat)
    
    # Resize style statistics to match content spatial dimensions if needed
    if style_mean.shape[2:] != content_mean.shape[2:]:
        style_mean = F.interpolate(style_mean, size=content_mean.shape[2:], mode='nearest')
        style_std = F.interpolate(style_std, size=content_mean.shape[2:], mode='nearest')
    
    normalized_feat = (content_feat - content_mean) / content_std
    styled_feat = normalized_feat * style_std + style_mean
    
    return styled_feat
import torch as t
import torch.nn.functional as F


class JointScaledImage(t.nn.Module):
    """Duplcate a ``Tensor Image`` to a new tensor image of scaled size.
    """

    def __init__(self, new_img_sizes, new_seg_size):
        assert isinstance(new_img_sizes, (tuple, list)), "'new_img_sizes' must be a tuple or list of two required image sizes!"
        assert len(new_img_sizes) == 2, "'new_img_sizes' must contain 2 items!"
        super().__init__()
        
        self.new_img_sizes = new_img_sizes
        self.new_seg_size = new_seg_size

    def forward(self, img:t.Tensor, seg):
        """
        Args:
            img (Tensor Image): Image tensor to scale.
            seg: Segmentation map

        Returns:
            (Tensor, ...), seg: Tuple of resized image tensors sames as the number of 'new_sizes' and unaltered segmentation mask.
        """
        # CAUTION: 'F.interpolate()' requires its input to be of 4-dimensions and hence the 'unsqueeze()'
        img_scaled1 = t.squeeze(F.interpolate(t.unsqueeze(img, dim=0), size=self.new_img_sizes[0], mode='bilinear', align_corners=True), dim=0)
        img_scaled2 = t.squeeze(F.interpolate(t.unsqueeze(img, dim=0), size=self.new_img_sizes[1], mode='bilinear', align_corners=True), dim=0)
        seg_scaled = t.squeeze(t.squeeze(F.interpolate(t.unsqueeze(t.unsqueeze(seg, dim=0), dim=0), size=self.new_seg_size, mode='nearest'), dim=0), dim=0)

        return (img_scaled1, img_scaled2),\
               (seg_scaled, seg)
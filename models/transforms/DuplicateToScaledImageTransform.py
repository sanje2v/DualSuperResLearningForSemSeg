import torch as t
import torch.nn.functional as F

class DuplicateToScaledImageTransform(t.nn.Module):
    """Duplcate a ``Tensor Image`` to a new tensor image of scaled size.
    """

    def __init__(self, new_size):
        super().__init__()
        self.new_size = new_size

    def forward(self, pic_tensor):
        """
        Args:
            pic_tensor (Tensor Image): Image tensor to scale.

        Returns:
            (Tensor, Tensor): (Scaled image tensor, Original image tensor).
        """
        scaled_pic_tensor = t.unsqueeze(pic_tensor, dim=0)
        scaled_pic_tensor = F.interpolate(scaled_pic_tensor, size=self.new_size, mode='bilinear', align_corners=True)
        scaled_pic_tensor = t.squeeze(scaled_pic_tensor, dim=0)

        return scaled_pic_tensor, pic_tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'
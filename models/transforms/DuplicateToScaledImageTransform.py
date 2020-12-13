import torch as t
import torch.nn.functional as F

class DuplicateToScaledImageTransform:
    """Duplcate a ``Tensor Image`` to a new tensor image of scaled size.
    """

    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, pic_tensor):
        """
        Args:
            pic_tensor (Tensor Image): Image tensor to scale.

        Returns:
            (Tensor, Tensor): (Scaled image tensor, Original Image tensor).
        """
        scaled_pic_tensor = t.squeeze(F.interpolate(t.unsqueeze(pic_tensor, dim=0), size=self.new_size), dim=0)
        
        return scaled_pic_tensor, pic_tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'
import torch as t
import torch.nn.functional as F

class DuplicateToScaledImageTransform(t.nn.Module):
    """Duplcate a ``Tensor Image`` to a new tensor image of scaled size.
    """

    def __init__(self, new_sizes):
        assert isinstance(new_sizes, (tuple, list)), "BUG CHECK: 'new_sizes' must be a tuple or list."
        assert len(new_sizes) == 2, "BUG CHECK: 'new_sizes' must be of length 2."

        super().__init__()
        self.new_sizes = new_sizes

    def forward(self, pic_tensor):
        """
        Args:
            pic_tensor (Tensor Image): Image tensor to scale.

        Returns:
            (Tensor, ...): Tuple of resized image tensors sames as the number of 'new_sizes'.
        """
        pic_tensor = t.unsqueeze(pic_tensor, dim=0)
        return [t.squeeze(F.interpolate(pic_tensor, size=self.new_sizes[i], mode='bilinear', align_corners=True), dim=0) for i in range(len(self.new_sizes))]

    def __repr__(self):
        return self.__class__.__name__ + '()'
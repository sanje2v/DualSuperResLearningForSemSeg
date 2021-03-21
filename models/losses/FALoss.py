import torch as t
import torch.nn.functional as F


class FALoss(t.nn.modules.loss._Loss):
    __constants__ = ['reduction']

    @staticmethod
    def _calculate_matrix_similarity(feature_map):
        feature_map_normalized = t.div(feature_map, t.linalg.norm(feature_map, ord=2, dim=(2, 3), keepdims=True))
        return t.matmul(t.transpose(feature_map_normalized, dim0=2, dim1=3), feature_map_normalized)


    def __init__(self, subsample_factor:int=8, size_average=None, reduce=None, reduction='mean') -> None:
        super().__init__(size_average=None, reduce=None, reduction=reduction)
        self.subsample_factor = subsample_factor

    def forward(self, feature_map1: t.Tensor, feature_map2: t.Tensor) -> t.Tensor:
        t.Assert(len(feature_map1.shape) == 4, "BUG CHECK: Feature map inputs to FALoss.forward() must have 4 dimensions (B, C, H, W).")
        t.Assert(feature_map1.shape == feature_map2.shape, "BUG CHECK: Feature map inputs to FALoss.forward() should be of same size.")

        # Subsample feature map and then calculate matrix similarity
        S_feature_map1 = FALoss._calculate_matrix_similarity(t.nn.AvgPool2d(self.subsample_factor)(feature_map1))
        S_feature_map2 = FALoss._calculate_matrix_similarity(t.nn.AvgPool2d(self.subsample_factor)(feature_map2))

        # Create repeats of matrix similarity so that we can calculate L1 norm between each element of one matrix to every other element
        S_feature_map1 = t.flatten(S_feature_map1, start_dim=2, end_dim=3)
        S_feature_map1 = t.repeat_interleave(S_feature_map1, repeats=S_feature_map1.shape[-1], dim=2)
        S_feature_map2 = t.flatten(S_feature_map2, start_dim=2, end_dim=3)
        S_feature_map2 = S_feature_map2.repeat(1, 1, S_feature_map2.shape[-1])

        return F.l1_loss(S_feature_map1,
                         S_feature_map2,
                         reduction=self.reduction)
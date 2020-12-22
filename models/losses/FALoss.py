import torch as t
import torch.nn.functional as F


class FALoss(t.nn.modules.loss._Loss):
    __constants__ = ['reduction']

    @staticmethod
    def _calculate_matrix_similarity(feature_map):
        feature_map_normalized = feature_map / t.linalg.norm(feature_map, ord=2, dim=(2, 3), keepdims=True)
        return t.matmul(t.transpose(feature_map_normalized, dim0=2, dim1=3), feature_map_normalized)


    def __init__(self, size_average=None, reduce=None, reduction='mean') -> None:
        return super().__init__(size_average=None, reduce=None, reduction=reduction)

    def forward(self, feature_map1: t.Tensor, feature_map2: t.Tensor) -> t.Tensor:
        t.Assert(len(feature_map1.shape) == 4, "BUG CHECK: Feature map inputs to FALoss.forward() must have 4 dimensions (B, C, H, W).")
        t.Assert(feature_map1.shape == feature_map2.shape, "BUG CHECK: Feature map inputs to FALoss.forward() should be of same size.")

        S_feature_map1 = FALoss._calculate_matrix_similarity(feature_map1)
        S_feature_map2 = FALoss._calculate_matrix_similarity(feature_map2)

        return F.l1_loss(S_feature_map1,
                         S_feature_map2,
                         reduction=self.reduction)
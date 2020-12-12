import torch as t


class FALoss(t.nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean') -> None:
        return super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, input: t.Tensor, target: t.Tensor) -> t.Tensor:
        pass
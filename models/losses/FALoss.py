from torch.nn.modules.loss import _Loss


class FALoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean') -> None:
        return super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        pass
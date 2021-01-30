import torch as t


class BaseModel(t.nn.Module):
    def __init__(self, profiler:t.autograd.profiler.profile=None):
        super().__init__()

        self.profiler = profiler

    def initialize_with_pretrained_weights(self, weights_dir, map_location=t.device('cpu')):
        raise NotImplementedError("Derived class of 'BaseModel' hasn't implemented 'initialize_with_pretrained_weights()' function.")
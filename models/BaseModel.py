import torch as t


class BaseModel(t.nn.Module):
    def initialize_with_pretrained_weights(self, weights_dir, map_location=t.device('cpu')):
        raise NotImplementedError("Derived class of 'BaseModel' hasn't implemented 'initialize_with_pretrained_weights()' function.")
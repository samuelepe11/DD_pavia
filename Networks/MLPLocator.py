# Import packages
import torch.nn as nn


# Classes
class MLPLocator(nn.Module):
    # Define attributes
    input_dim = 768

    def __init__(self, params, device="cpu"):
        super(MLPLocator, self).__init__()
        self.device = device
        self.params = params

        # Define modules
        self.fc_sizes = None
        self.fc = None
        self.initialize_modules(params)

    def initialize_modules(self, params):
        # Define final fully connected layers
        self.fc_sizes = [self.input_dim] + [params["n_fc_neurons"]] * params["n_fc_layers"]
        self.initialize_fc_module()

    def initialize_fc_module(self):
        fc_list = []
        for i in range(len(self.fc_sizes) - 1):
            fc_list.append(nn.Linear(self.fc_sizes[i], self.fc_sizes[i + 1]))
            fc_list.append(nn.ReLU())
        fc_list.append(nn.Linear(self.fc_sizes[-1], 1))
        fc_list.append(nn.Sigmoid())
        self.fc = nn.Sequential(*fc_list)

    def forward(self, x):
        out = self.fc(x)
        return out

    def set_training(self, training=True):
        if training:
            self.train()
        else:
            self.eval()

    def set_cuda(self, cuda=True):
        if cuda:
            self.cuda()
        else:
            self.cpu()

        # Set specific layers
        for layer_name, _ in self.named_children():
            if cuda:
                getattr(self, layer_name).cuda()
            else:
                getattr(self, layer_name).cpu()

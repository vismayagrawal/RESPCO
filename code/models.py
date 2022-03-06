import torch.nn as nn
import torch.nn.functional as F
import torch

class linear_without_tanh(nn.Module):
    '''Just plain linear regression'''
    def __init__(self, in_channels=1, out_channels=1):
        super(linear_without_tanh, self).__init__()
        kernel_size = 1
        model = [nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        # input size: (B, C, W)
        return self.model(x)


class linear_with_tanh(nn.Module):
    '''Linear regression after applying tanh operator to respiration'''
    def __init__(self, in_channels=1, out_channels=1):
        super(linear_with_tanh, self).__init__()
        kernel_size = 1
        model = [nn.Tanh(),
                 nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        # input size: (B, C, W)
        return self.model(x)


# 1 layer NN
class conv_1_layer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(conv_1_layer, self).__init__()
        kernel_size = 7 # keep an odd number
        padding = kernel_size // 2
        model = [nn.Tanh(),
                 nn.ReplicationPad1d(padding),
                 nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        # input size: (B, C, W)
        return self.model(x)

# 2 layer NN
class conv_2_layer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(conv_2_layer, self).__init__()
        kernel_size = 7
        model = [nn.Tanh(),
                 nn.Conv1d(in_channels, 4, kernel_size,stride=2),
                 nn.InstanceNorm1d(4),
                 nn.ReLU(),
                 nn.ConvTranspose1d(4, out_channels, kernel_size, stride=2, output_padding=1)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        # input size: (B, C, W)
        return self.model(x)

# 4 layer NN
class conv_4_layer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(conv_4_layer, self).__init__()
        kernel_size = 7
        # Initial convolution block
        model = [nn.Tanh(),
                 nn.Conv1d(in_channels, 4, kernel_size,stride=2),
                 nn.InstanceNorm1d(4),
                 nn.ReLU(),
                 nn.Conv1d(4, 8, kernel_size, stride=2),
                 nn.InstanceNorm1d(8),
                 nn.ReLU(),
                 nn.ConvTranspose1d(8, 4, kernel_size, stride=2),
                 nn.InstanceNorm1d(4),
                 nn.ReLU(),
                 nn.ConvTranspose1d(4, out_channels, kernel_size, stride=2, output_padding=1),
                ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        # input size: (B, C, W)
        return self.model(x)


#conv 6 layer
class conv_6_layer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(conv_6_layer, self).__init__()
        kernel_size = 7
        # Initial convolution block
        model = [nn.Tanh(),
                 nn.Conv1d(in_channels, 4, kernel_size,stride=2),
                 nn.InstanceNorm1d(4),
                 nn.ReLU(),
                 nn.Conv1d(4, 8, kernel_size, stride=2),
                 nn.InstanceNorm1d(8),
                 nn.ReLU(),
                 nn.Conv1d(8, 16, kernel_size, stride=2),
                 nn.InstanceNorm1d(16),
                 nn.ReLU(),
                 nn.ConvTranspose1d(16, 8, kernel_size, stride=2,output_padding=1),
                 nn.InstanceNorm1d(8),
                 nn.ReLU(),
                 nn.ConvTranspose1d(8, 4, kernel_size, stride=2),
                 nn.InstanceNorm1d(4),
                 nn.ReLU(),
                 nn.ConvTranspose1d(4, out_channels, kernel_size, stride=2, output_padding=1),
                ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        # input size: (B, C, W)
        return self.model(x)

class conv_multi_layer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(conv_multi_layer, self).__init__()

        in_features = 4
        # Initial convolution block
        model = [   nn.ReflectionPad1d(3),
                    nn.Conv1d(in_channels, in_features, 7),
                    nn.InstanceNorm1d(in_features),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv1d(in_features, out_features, 7, stride=2, padding=1),
                        nn.InstanceNorm1d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose1d(in_features, out_features, 7, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm1d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [ 
                    nn.ReflectionPad1d(2),
                    nn.Conv1d(in_features, out_channels, 7)]
                    # nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
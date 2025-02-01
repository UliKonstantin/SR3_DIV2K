from torch import nn, optim
from EncoderDecoder import *
# Encoder Block for downsampling

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, time_steps, activation = "relu"):
        super().__init__()
        self.conv = conv_block(in_c, out_c, time_steps = time_steps, activation = activation, embedding_dims = out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs, time = None):
        x = self.conv(inputs, time)
        p = self.pool(x)
        return x, p

# Decoder Block for upsampling
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, time_steps, activation = "relu"):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c, time_steps = time_steps, activation = activation, embedding_dims = out_c)
    def forward(self, inputs, skip, time = None):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x, time)
        return x
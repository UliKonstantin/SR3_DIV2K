from kornia.contrib.models.sam.architecture.transformer import Attention
from torch import nn, optim
from Model.EncoderDecoder import *
from Model.Attention import *

class UNet(nn.Module):

    def __init__(self, input_channels = 3, output_channels = 3, time_steps = 512):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.time_steps = time_steps
        self.time_steps = time_steps

        self.e1 = encoder_block(self.input_channels, 64, time_steps=self.time_steps)
        self.e2 = encoder_block(64, 128, time_steps=self.time_steps)
        # self.da2 = AttnBlock(128)
        self.e3 = encoder_block(128, 256, time_steps=self.time_steps)
        self.da3 = AttnBlock(256)
        self.e4 = encoder_block(256, 512, time_steps=self.time_steps)
        self.da4 = AttnBlock(512)

        self.b = conv_block(512, 1024, time_steps=self.time_steps) # bottleneck
        self.ba1 = AttnBlock(1024)
        self.d1 = decoder_block(1024, 512, time_steps=self.time_steps)
        self.ua1 = AttnBlock(512)
        self.d2 = decoder_block(512, 256, time_steps=self.time_steps)
        self.ua2 = AttnBlock(256)
        self.d3 = decoder_block(256, 128, time_steps=self.time_steps)
        # self.ua3 = AttnBlock(128)
        self.d4 = decoder_block(128, 64, time_steps=self.time_steps)
        # self.ua4 = AttnBlock(64)
        self.outputs = nn.Conv2d(64, self.output_channels, kernel_size=1, padding=0)

    def forward(self, inputs, t = None):
        # downsampling block
        s1, p1 = self.e1(inputs, t)
        s2, p2 = self.e2(p1, t)
        s3, p3 = self.e3(p2, t)
        p3 = self.da3(p3)
        s4, p4 = self.e4(p3, t)
        p4 = self.da4(p4)
        # bottleneck
        b = self.b(p4, t)
        b = self.ba1(b)
        # upsampling block
        d1 = self.d1(b, s4, t)
        d1 = self.ua1(d1)
        d2 = self.d2(d1, s3, t)
        d2 = self.ua2(d2)
        d3 = self.d3(d2, s2, t)
        d4 = self.d4(d3, s1, t)
        outputs = self.outputs(d4)
        return outputs
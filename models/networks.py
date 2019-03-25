import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.autograd import grad
import torch.nn.functional as F

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks_unit':
        netG = ResnetGeneratorUnit(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_9blocks_unit_shading':
        netG = ResnetGeneratorUnitShading(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    return netG


def utils_cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]


class ResnetGeneratorUnit(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGeneratorUnit, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model0 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model0 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        model1 = [nn.Conv2d(2, int(ngf * mult),
                                         kernel_size=3, stride=1,
                                         padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult)),
                      nn.ReLU(True)]


        model = []
        model += [nn.Conv2d(ngf * mult*2, int(ngf * mult),
                                         kernel_size=3, stride=1,
                                         padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult)),
                      nn.ReLU(True)]
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]


        upsample2 = nn.Upsample(scale_factor=2)
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [upsample2, nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=1,
                                         padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
            

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model0 = nn.Sequential(*model0)
        self.model1 = nn.Sequential(*model1)
        self.model = nn.Sequential(*model)


    def forward(self, input1, input2):
        f1 = self.model0(input1)
        f2 = self.model1(input2)
        f1 = torch.cat((f1,f2),1)
        out = self.model(f1)

        return out


class ResnetGeneratorUnitShading(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGeneratorUnitShading, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # edge
        model0 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model0 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        
        # unit
        mult = 2**n_downsampling
        model1 = [nn.Conv2d(4, int(ngf * mult),
                                         kernel_size=3, stride=1,
                                         padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult)),
                      nn.ReLU(True)]

        # abs
        model2 = [nn.Conv2d(input_nc, ngf, kernel_size=7,
                                stride=4, padding=2, bias=use_bias),
                      norm_layer(ngf),
                      nn.ReLU(True)]
        model2 += [nn.Conv2d(ngf, ngf * 4, kernel_size=7,
                                stride=4, padding=2, bias=use_bias),
                      norm_layer(ngf * 4),
                      nn.ReLU(True)]
        upsample2 = nn.Upsample(scale_factor=4)
        model2 += [upsample2]



        # combine
        model = [nn.Conv2d(ngf * 4*3, int(ngf * 4),
                                         kernel_size=3, stride=1,
                                         padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * 4)),
                      nn.ReLU(True)]
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]


        upsample2 = nn.Upsample(scale_factor=2)
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [upsample2, nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=1,
                                         padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
            

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model0 = nn.Sequential(*model0)
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model = nn.Sequential(*model)


    def forward(self, input1, input2, input3):
        f1 = self.model0(input1)
        f2 = self.model1(input2)
        f3 = self.model2(input3)
        f1 = torch.cat((f1,f2,f3),1)
        out = self.model(f1)

        return out


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

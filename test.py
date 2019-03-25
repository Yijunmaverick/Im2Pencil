import os
import torch
from options.test_options import TestOptions
from models import create_model
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from scipy.ndimage import filters
from pylab import *
import glob
from models import networks


def preprocess(img_path, num_style, ind):
    im = Image.open(img_path)
    w = im.size[0]
    h = im.size[1]

    factor = 16
    h = int(round(h*1.0/factor)*factor)
    w = int(round(w*1.0/factor)*factor)

    result = im.resize((w, h), Image.BICUBIC)
    result = np.reshape(result, (h, w, 1))
    result = result.repeat(3, axis=2)

    unit = torch.zeros(num_style, int(h/4), int(w/4))
    unit[ind].fill_(1)

    return result, unit

def XDoG(im, Sigma):

    Gamma = 0.99
    Phi = 200
    Epsilon = 0.1
    k = 1.6

    im = im[:,:,0]
    h = im.shape[0]
    w = im.shape[1]

    im2 = filters.gaussian_filter(im, Sigma)
    im3 = filters.gaussian_filter(im, Sigma* k)
    differencedIm2 = im2 - (Gamma * im3)
    for i in range(h):
        for j in range(w):
            if differencedIm2[i, j] >= Epsilon:
                differencedIm2[i, j] = 255
            else:
                differencedIm2[i, j] = 1 + tanh(Phi * (differencedIm2[i, j]-Epsilon))

    result = differencedIm2.astype(np.uint8)
    result = np.reshape(result, (result.shape[0], result.shape[1], 1))
    result = result.repeat(3, axis=2)

    return result

def deprocess(output):
    output = np.asarray(output[0].data.cpu().float())
    output = output.transpose(1, 2, 0)
    output = output*0.5 + 0.5
    output_tosave = Image.fromarray((output*255).astype(np.uint8))
    return output, output_tosave



def main():
    in_dir = 'extract_edge_tone/data/edge_tone/'
    out_dir = 'output'
    if not os.path.exists(out_dir): os.mkdir(out_dir)

    opt = TestOptions().parse()
    opt.which_model_netG = 'resnet_9blocks_unit' 
    opt.model = 'pix2pix2_twobranch'
    opt.pretrained_model = './pretrained_models/edge_model/latest_net_G.pth'
    netG1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type)
    netG1.load_state_dict(torch.load(opt.pretrained_model))
    netG1.cuda()

    opt.which_model_netG = 'resnet_9blocks_unit_shading'
    opt.model = 'pix2pix3_twobranch'
    opt.pretrained_model = './pretrained_models/shading_model/latest_net_G.pth'
    netG2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type)
    netG2.load_state_dict(torch.load(opt.pretrained_model))
    netG2.cuda()


    for files in glob.glob(in_dir + '/*_edge.jpg'):
        filepath, filename = os.path.split(files)
        print(filename)

        imgE, unitE = preprocess(files, 2, opt.outline_style)
        unitE = unitE.unsqueeze(0)

        filenameS = filename[:-9] + '_gf.jpg'
        imgS, unitS = preprocess(os.path.join(in_dir, filenameS), 4, opt.shading_style)
        unitS = unitS.unsqueeze(0)
        
        imgE_xdog = XDoG(imgE, opt.Sigma)
        imgE_xdog = transforms.ToTensor()(imgE_xdog).unsqueeze(0)
        imgE_xdog = (imgE_xdog - 0.5)/0.5
        with torch.no_grad():
            outputE = netG1(Variable(imgE_xdog).cuda(), Variable(unitE).cuda())
        outputE, outputE_tosave = deprocess(outputE)
        outputE_tosave.save(os.path.join(out_dir, filename[:-9]+'_e'+str(opt.outline_style)+'.png'))      


        imgS = transforms.ToTensor()(imgS).unsqueeze(0)
        imgS = (imgS - 0.5)/0.5
        imgE = transforms.ToTensor()(imgE).unsqueeze(0)
        imgE = (imgE - 0.5)/0.5
        with torch.no_grad():
            outputS = netG2(Variable(imgE).cuda(), Variable(unitS).cuda(), Variable(imgS).cuda())
        outputS, outputS_tosave = deprocess(outputS)
        outputS_tosave.save(os.path.join(out_dir, filename[:-9]+'_s'+str(opt.shading_style)+'.png'))

        output = outputE * outputS
        output_tosave = Image.fromarray((output*255).astype(np.uint8))
        output_tosave.save(os.path.join(out_dir, filename[:-9]+'_e'+str(opt.outline_style)+'_s'+str(opt.shading_style)+'_combo.png'))


if __name__ == '__main__':
    main()
    print('done!')







        

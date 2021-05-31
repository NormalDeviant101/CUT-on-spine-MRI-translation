import numpy as np
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from data.unaligned_dataset import UnalignedDataset
# generate random integer values
from random import seed
import torch
from torchsummary import summary

seed(1)


class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample', 'global_pool', 'strided_conv'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        #bs_per_gpu = 1
        print(self.real_A.size(0))
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        ######################################################################################################

        ######################################################################################################
        #input['S1_paths'] = UnalignedDataset.__getitem__(self, index)['S1_paths']
        #input['S2_paths'] = UnalignedDataset.__getitem__(self, index)['S2_paths']
        #input['S3_paths'] = UnalignedDataset.__getitem__(self, index)['S3_paths']
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)

        if self.opt.phase == "train":
            self.mask_A = input['mask_S1' if AtoB else 'mask_T'].to(self.device)
            self.mask_B = input['mask_T' if AtoB else 'mask_S1'].to(self.device)

        if AtoB:
            img_list = ['S1_paths', 'S2_paths', 'S3_paths']
            self.image_paths = (input[key] for key in img_list)
            self.image_paths = ' '.join(str(path) for path in self.image_paths)
            self.image_paths_save = (input[key] for key in img_list)
            self.image_paths_save = ' '.join(str(path) for path in self.image_paths_save)

            if self.opt.phase == "train":
                mask_list = ['mask_S1_paths']
                self.mask_paths = (input[key] for key in mask_list)
                self.mask_paths = ' '.join(str(path) for path in self.mask_paths)
                self.mask_paths_save = (input[key] for key in mask_list)
                self.mask_paths_save = ' '.join(str(path) for path in self.mask_paths_save)

        else:
            self.image_paths = input['B_paths']
            self.image_paths = ' '.join(str(path) for path in self.image_paths)
            self.image_paths_save = input['B_paths']
            self.image_paths_save = ' '.join(str(path) for path in self.image_paths_save)

            if self.opt.phase == "train":
                mask_list = ['mask_T_paths']
                self.mask_paths = (input[key] for key in mask_list)
                self.mask_paths = ' '.join(str(path) for path in self.mask_paths)
                self.mask_paths_save = (input[key] for key in mask_list)
                self.mask_paths_save = ' '.join(str(path) for path in self.mask_paths_save)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        real_B_broadcast = self.real_B.expand(-1, 3, -1, -1)
        self.real = torch.cat((self.real_A, real_B_broadcast), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        self.fake_B = self.fake_B.expand(-1, 3, -1, -1)
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss_mask(self):
        # Loss of the Mask_Fake(Mask_A) multiply with Fake Image ; Masked Fake.
        # print('The size of mask A is: ', self.mask_A.shape)
        # print('The size of fake B is: ', self.fake_B.shape)
        fake = self.fake.detach()
        self.mask_fake = self.netD(self.mask_A * fake)
        loss_D_mask_fake = self.criterionGAN(self.mask_fake, False)
        self.loss_D_mask_fake = loss_D_mask_fake.mean()

        # Loss of the Mask_Real(Mask_B) multiply with Real Image ; Masked Real.
        self.mask_real = self.netD(self.mask_B * self.real_B)
        loss_D_mask_real = self.criterionGAN(self.mask_real, True)
        self.loss_D_mask_real = loss_D_mask_real.mean()

        # Calculate the loss of the mask.
        self.loss_D_mask = (self.loss_D_mask_real + self.loss_D_mask_fake) * 0.5

        return self.loss_D_mask

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()

        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D = self.loss_D + self.compute_D_loss_mask() * 0.5
        return self.loss_D


    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(self.fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:

            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B.expand(-1, 3, -1, -1), self.idt_B.expand(-1, 3, -1, -1))
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):

        n_layers = len(self.nce_layers)
        #n_layers = 1
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        # print("feat_k shape is : ", list(map(lambda x: x.shape, feat_k)))
        # print("netF is: ", self.netF)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def get_real_A(self):
        return self.real_A

    def get_real_B(self):
        return self.real_B

    def get_fake_B(self):
        return self.fake_B

    # def get_model_summary(self):
    #     return summary(self, (3, 896, 896))


# # summarize model
# from collections import OrderedDict
# import pandas as pd
# import torch
# from torch import nn as nn
# from torch.autograd import Variable
# import torch.nn.modules.module
#
# class TorchSummarizeDf(object):
#     def __init__(self, model, weights=False, input_shape=True, nb_trainable=False, debug=False):
#         """
#         Summarizes torch model by showing trainable parameters and weights.
#         author: wassname
#         url: https://gist.github.com/wassname/0fb8f95e4272e6bdd27bd7df386716b7
#         license: MIT
#         Modified from:
#         - https://github.com/pytorch/pytorch/issues/2001#issuecomment-313735757
#         - https://gist.github.com/wassname/0fb8f95e4272e6bdd27bd7df386716b7/
#         Usage:
#             import torchvision.models as models
#             model = models.alexnet()
#             # attach temporary hooks using `with`
#             with TorchSummarizeDf(model) as tdf:
#                 x = Variable(torch.rand(2, 3, 224, 224))
#                 y = model(x)
#                 df = tdf.make_df()
#             print(df)
#             # Total parameters 61100840
#             #              name class_name        input_shape       output_shape  nb_params
#             # 1     features=>0     Conv2d  (-1, 3, 224, 224)   (-1, 64, 55, 55)      23296
#             # 2     features=>1       ReLU   (-1, 64, 55, 55)   (-1, 64, 55, 55)          0
#             # ...
#         """
#         # Names are stored in parent and path+name is unique not the name
#         self.names = get_names_dict(model)
#
#         # store arguments
#         self.model = model
#         self.weights = weights
#         self.input_shape = input_shape
#         self.nb_trainable = nb_trainable
#         self.debug = debug
#
#         # create properties
#         self.summary = OrderedDict()
#         self.hooks = []
#
#     def register_hook(self, module):
#         """Register hooks recursively"""
#         self.hooks.append(module.register_forward_hook(self.hook))
#
#     def hook(self, module, input, output):
#         """This hook is applied when each module is run"""
#         class_name = str(module.__class__).split('.')[-1].split("'")[0]
#         module_idx = len(self.summary)
#
#         name = None
#         for key, item in self.names.items():
#             if item == module:
#                 name = key
#         if name is None:
#             name = '{}_{}'.format(class_name, module_idx)
#
#         m_key = module_idx + 1
#
#         self.summary[m_key] = OrderedDict()
#         self.summary[m_key]['name'] = name
#         self.summary[m_key]['class_name'] = class_name
#
#         # Handle multiple inputs
#         if self.input_shape:
#             # for each input remove batch size and replace with one
#             self.summary[m_key][
#                 'input_shape'] = format_input_output_shape(input)
#
#         # Handle multiple outputs
#         self.summary[m_key]['output_shape'] = format_input_output_shape(output)
#
#         if self.weights:
#             self.summary[m_key]['weights'] = list(
#                 [tuple(p.size()) for p in module.parameters()])
#
#         if self.nb_trainable:
#             self.summary[m_key]['nb_trainable'] = get_params(module, True)
#
#         self.summary[m_key]['nb_params'] = get_params(module, True)
#
#         if self.debug:
#             print(self.summary[m_key])
#
#     def __enter__(self):
#
#         # register hook
#         self.model.apply(self.register_hook)
#
#         # make a forward pass
#         self.training = self.model.training
#         if self.training:
#             self.model.eval()
#
#         return self
#
#     def make_df(self):
#         """Make dataframe."""
#         df = pd.DataFrame.from_dict(self.summary, orient='index')
#
#         df['level'] = df['name'].apply(lambda name: name.count('.'))
#
#         total_params = get_params(self.model, False)
#         total_trainable_params = get_params(self.model, True)
#         print('Total parameters', total_params)
#         print('Total trainable parameters', total_trainable_params)
#         return df
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#
#         if exc_type or exc_val or exc_tb:
#             # to help with debugging your model lets print the summary even if it fails
#             df_summary = pd.DataFrame.from_dict(self.summary, orient='index')
#             print(df_summary)
#
#         if self.training:
#             self.model.train()
#
#         # remove these hooks
#         for h in self.hooks:
#             h.remove()
#
#
#
#
# def get_names_dict(model):
#     """Recursive walk to get names including path."""
#     names = {}
#
#     def _get_names(module, parent_name=''):
#         for key, module in module.named_children():
#             name = parent_name + '.' + key if parent_name else key
#             names[name] = module
#             if isinstance(module, torch.nn.Module):
#                 _get_names(module, parent_name=name)
#
#     _get_names(model)
#     return names
#
#
# def get_params(module, nb_trainable=False):
#     if nb_trainable:
#         params = sum([torch.LongTensor(list(p.size())).prod() for p in module.parameters() if p.requires_grad])
#     else:
#         params = sum([torch.LongTensor(list(p.size())).prod() for p in module.parameters()])
#     if isinstance(params, torch.Tensor):
#         params = params.item()
#     return params
#
#
# def format_input_output_shape(tensors):
#     "Recursively get N nested levels of inputs."""
#
#     def _format_input_output_shape(tensors):
#         if isinstance(tensors, (list, tuple)):
#             if len(tensors) == 1:
#                 return _format_input_output_shape(tensors[0])
#             else:
#                 return [_format_input_output_shape(tensor) for tensor in tensors]
#         else:
#             return [(-1,) + tuple(tensors.size())[1:]]
#
#     return _format_input_output_shape(tensors)
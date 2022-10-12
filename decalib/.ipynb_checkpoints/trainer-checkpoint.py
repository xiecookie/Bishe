# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from time import time
from skimage.io import imread
import cv2
import pickle
from loguru import logger
from datetime import datetime
from tqdm import tqdm

from .utils.renderer import SRenderY
from .models.encoders import ResnetEncoder
from .models.FLAME import FLAME, FLAMETex
from .models.decoders import Generator
from .utils import util
from .utils.rotation_converter import batch_euler2axis
from .datasets import datasets
from .utils.config import cfg

torch.backends.cudnn.benchmark = True
from .utils import lossfunc
from .datasets import build_datasets


class Trainer(object):
    def __init__(self, model, config=None, device='cuda:0'):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.batch_size = self.cfg.dataset.batch_size
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size
        self.train_detail = self.cfg.train.train_detail

        # deca model
        self.deca = model.to(self.device)
        self.configure_optimizers()
        self.load_checkpoint()

        # initialize loss  
        # # initialize loss
        self.mrf_loss = lossfunc.IDMRFLoss()

        logger.add(os.path.join(self.cfg.output_dir, self.cfg.train.log_dir, 'train.log'))
        if self.cfg.train.write_summary:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.output_dir, self.cfg.train.log_dir))

    def configure_optimizers(self):
        self.opt = torch.optim.Adam(
            list(self.deca.E_detail.parameters()) + \
            list(self.deca.D_detail.parameters()),
            lr=self.cfg.train.lr,
            amsgrad=False)


    def load_checkpoint(self):
        model_dict = self.deca.model_dict()
        # resume training, including model weight, opt, steps
        # import ipdb; ipdb.set_trace()
        if self.cfg.train.resume and os.path.exists(os.path.join(self.cfg.output_dir, 'model.tar')):
            checkpoint = torch.load(os.path.join(self.cfg.output_dir, 'model.tar'))
            for key in model_dict.keys():
                if key in checkpoint.keys():
                    util.copy_state_dict(model_dict[key], checkpoint[key])
            util.copy_state_dict(self.opt.state_dict(), checkpoint['opt'])
            self.global_step = checkpoint['global_step']
            logger.info(f"resume training from {os.path.join(self.cfg.output_dir, 'model.tar')}")
            logger.info(f"training start from step {self.global_step}")
        # load model weights only
        elif os.path.exists(self.cfg.pretrained_modelpath):
            checkpoint = torch.load(self.cfg.pretrained_modelpath)
            key = 'E_flame'
            util.copy_state_dict(model_dict[key], checkpoint[key])
            self.global_step = 0
        else:
            logger.info('model path not found, start training from scratch')
            self.global_step = 0

    def training_step(self, batch):
        self.deca.train()
        # [B, K, 3, size, size] ==> [BxK, 3, size, size]
        # images = batch['image'].to(self.device);
        # images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        ftex224 = batch['ftex224'].to(self.device)
        ftex256 = batch['ftex256'].to(self.device)
        lightcode = batch['light'].to(self.device)
        gttex = batch['gttex'].to(self.device)
        trans_verts = batch['verts'].to(self.device)

        # -- encoder include <detail, ftex>
        codedict = self.deca.encode(ftex224, use_detail=self.train_detail)

        batch_size = ftex256.shape[0]

        # -- decoder
        detailcode = codedict['detail']
        # verts = codedict['verts']

        # trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

        # ------ rendering
        ops = self.deca.render(trans_verts, trans_verts, gttex, None)

        uv_z = self.deca.D_detail(detailcode)
        # render detail
        uv_detail_normals = self.deca.displacement2normal(uv_z, trans_verts, ops['normals'])
        uv_shading = self.deca.render.add_SHlight(uv_detail_normals, lightcode.detach())
        uv_texture = ftex256.detach() * uv_shading
        gt_uv_texture = gttex.detach() * uv_shading
        # predicted_detail_images = F.grid_sample(uv_texture, ops['grid'].detach(), align_corners=False)

        #### ----------------------- Losses
        losses = {}
        ############################### details
        pi = 0
        new_size = 224

        losses['tex'] = (gt_uv_texture - uv_texture).abs().mean() * self.cfg.loss.photo_D

        losses['z_reg'] = torch.mean(uv_z.abs()) * self.cfg.loss.reg_z
        losses['z_diff'] = lossfunc.shading_smooth_loss(uv_shading) * self.cfg.loss.reg_diff
        if self.cfg.loss.reg_sym > 0.:
            losses['z_sym'] = ((
                        uv_z - torch.flip(uv_z, [-1]).detach()).abs()).sum() * self.cfg.loss.reg_sym
        opdict = {
            'trans_verts': trans_verts,
            'ftex': ftex256
        }

        #########################################################
        all_loss = 0.
        losses_key = losses.keys()
        for key in losses_key:
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
        return losses, opdict

    def validation_step(self):
        self.deca.eval()
        try:
            batch = next(self.val_iter)
        except:
            self.val_iter = iter(self.val_dataloader)
            batch = next(self.val_iter)
        images = batch['image'].to(self.device);
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        with torch.no_grad():
            codedict = self.deca.encode(images)
            opdict, visdict = self.deca.decode(codedict)
        savepath = os.path.join(self.cfg.output_dir, self.cfg.train.val_vis_dir, f'{self.global_step:08}.jpg')
        grid_image = util.visualize_grid(visdict, savepath, return_gird=True)
        self.writer.add_image('val_images', (grid_image / 255.).astype(np.float32).transpose(2, 0, 1), self.global_step)
        self.deca.train()

    def evaluate(self):
        ''' NOW validation 
        '''
        os.makedirs(os.path.join(self.cfg.output_dir, 'NOW_validation'), exist_ok=True)
        savefolder = os.path.join(self.cfg.output_dir, 'NOW_validation', f'step_{self.global_step:08}')
        os.makedirs(savefolder, exist_ok=True)
        self.deca.eval()
        # run now validation images
        from .datasets.now import NoWDataset
        dataset = NoWDataset(scale=(self.cfg.dataset.scale_min + self.cfg.dataset.scale_max) / 2)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False,
                                num_workers=8,
                                pin_memory=True,
                                drop_last=False)
        faces = self.deca.flame.faces_tensor.cpu().numpy()
        for i, batch in enumerate(tqdm(dataloader, desc='now evaluation ')):
            images = batch['image'].to(self.device)
            imagename = batch['imagename']
            with torch.no_grad():
                codedict = self.deca.encode(images)
                _, visdict = self.deca.decode(codedict)
                codedict['exp'][:] = 0.
                codedict['pose'][:] = 0.
                opdict, _ = self.deca.decode(codedict)
            # -- save results for evaluation
            verts = opdict['verts'].cpu().numpy()
            landmark_51 = opdict['landmarks3d_world'][:, 17:]
            landmark_7 = landmark_51[:, [19, 22, 25, 28, 16, 31, 37]]
            landmark_7 = landmark_7.cpu().numpy()
            for k in range(images.shape[0]):
                os.makedirs(os.path.join(savefolder, imagename[k]), exist_ok=True)
                # save mesh
                util.write_obj(os.path.join(savefolder, f'{imagename[k]}.obj'), vertices=verts[k], faces=faces)
                # save 7 landmarks for alignment
                np.save(os.path.join(savefolder, f'{imagename[k]}.npy'), landmark_7[k])
                for vis_name in visdict.keys():  # ['inputs', 'landmarks2d', 'shape_images']:
                    if vis_name not in visdict.keys():
                        continue
                    # import ipdb; ipdb.set_trace()
                    image = util.tensor2image(visdict[vis_name][k])
                    name = imagename[k].split('/')[-1]
                    # print(os.path.join(savefolder, imagename[k], name + '_' + vis_name +'.jpg'))
                    cv2.imwrite(os.path.join(savefolder, imagename[k], name + '_' + vis_name + '.jpg'), image)
            # visualize results to check
            util.visualize_grid(visdict, os.path.join(savefolder, f'{i}.jpg'))

        ## then please run main.py in https://github.com/soubhiksanyal/now_evaluation, it will take around 30min to get the metric results
        self.deca.train()

    def prepare_data(self):
        self.train_dataset = build_datasets.build_train(self.cfg.dataset)
        # self.val_dataset = build_datasets.build_val(self.cfg.dataset)
        logger.info('---- training data numbers: ', len(self.train_dataset))

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.cfg.dataset.num_workers,
                                           pin_memory=True,
                                           drop_last=True)
        self.train_iter = iter(self.train_dataloader)
        # self.val_dataloader = DataLoader(self.val_dataset, batch_size=8, shuffle=True,
        #                                  num_workers=8,
        #                                  pin_memory=True,
        #                                  drop_last=False)
        # self.val_iter = iter(self.val_dataloader)

    def fit(self):
        self.prepare_data()
        iters_every_epoch = int(len(self.train_dataset) / self.batch_size)
        start_epoch = self.global_step // iters_every_epoch
        for epoch in range(start_epoch, self.cfg.train.max_epochs):
            # for step, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch: {epoch}/{self.cfg.train.max_epochs}")):
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch + 1}/{self.cfg.train.max_epochs}]"):
                if epoch * iters_every_epoch + step < self.global_step:
                    continue
                try:
                    batch = next(self.train_iter)
                except:
                    self.train_iter = iter(self.train_dataloader)
                    batch = next(self.train_iter)
                losses, opdict = self.training_step(batch)
                if self.global_step % self.cfg.train.log_steps == 0:
                    loss_info = f"\n Epoch: {epoch}, Iter: {step}/{iters_every_epoch}, Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} \n"
                    for k, v in losses.items():
                        loss_info = loss_info + f'{k}: {v:.4f}, '
                        if self.cfg.train.write_summary:
                            self.writer.add_scalar('train_loss/' + k, v, global_step=self.global_step)
                    logger.info(loss_info)

                if self.global_step % self.cfg.train.vis_steps == 0:
                    # todo : add displacement visualization
                    pass

                if self.global_step > 0 and self.global_step % self.cfg.train.checkpoint_steps == 0:
                    model_dict = self.deca.model_dict()
                    model_dict['opt'] = self.opt.state_dict()
                    model_dict['global_step'] = self.global_step
                    model_dict['batch_size'] = self.batch_size
                    torch.save(model_dict, os.path.join(self.cfg.output_dir, 'model' + '.tar'))
                    # 
                    if self.global_step % self.cfg.train.checkpoint_steps * 10 == 0:
                        os.makedirs(os.path.join(self.cfg.output_dir, 'models'), exist_ok=True)
                        torch.save(model_dict,
                                   os.path.join(self.cfg.output_dir, 'models', f'{self.global_step:08}.tar'))

                # if self.global_step % self.cfg.train.val_steps == 0:
                #     self.validation_step()

                # if self.global_step % self.cfg.train.eval_steps == 0:
                #     self.evaluate()

                all_loss = losses['all_loss']
                self.opt.zero_grad();
                all_loss.backward();
                self.opt.step()
                self.global_step += 1
                if self.global_step > self.cfg.train.max_steps:
                    break

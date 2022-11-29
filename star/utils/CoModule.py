import torch.nn as nn
from coperception.utils.detection_util import *
from coperception.utils.min_norm_solvers import MinNormSolver
import numpy as np
import torch
import torch.nn.functional as F

# Juexiao add for mae -----
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import math
from coperception.utils.move_optim import optimizer_to
# -------------------------
from coperception.utils.CoDetModule import FaFModule


class CoModule(FaFModule):
    def __init__(self, model, optimizer, com):
        self.mae_loss_scaler = NativeScaler()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = None
        if com=="late" or com=="vqvae":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=[50, 100, 150, 200], gamma=0.5
            )

    def resume_from_cpu(self, checkpoint, device, trainable=True):
        """
        This function load state dict to model and optimizer on cpu, and move it back to device.
        This avoids a GPU memory surge issue.
        NOTE: assume checkpoint is loaded in cpu
        """
        # handles model
        self.model = self.model.cpu()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(device)
        if trainable:
            # handles optimizer
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            optimizer_to(self.optimizer, device)
            # possible extension: reinitialize scheduler based on this new optimizer
            self.scheduler = self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=[50, 100, 150, 200], gamma=0.5
            )

    # used by scene completion task
    def step_completion(self, data, batch_size, loss_fn='ce', trainable=False):
        bev_seq = data['bev_seq']
        trans_matrices = data['trans_matrices']
        num_agent = data['num_agent']

        result, ind_pred = self.model(bev_seq, trans_matrices, num_agent, batch_size=batch_size)

        loss_fn_dict = {
            'mse': nn.MSELoss(),
            'bce': nn.BCELoss(),
            'ce': nn.CrossEntropyLoss(),
            'l1': nn.L1Loss(),
            'smooth_l1': nn.SmoothL1Loss(),
        }

        loss = -1
        if trainable:
            # labels = data['bev_seq_teacher']
            # labels = labels.permute(0, 1, 4, 2, 3).squeeze()  # (Batch, seq, z, h, w)
            # loss = 10000 * loss_fn_dict[loss_fn](result, labels)
            target = bev_seq.permute(0, 1, 4, 2, 3).squeeze(1)
            target = target.type(torch.LongTensor).to(ind_pred.device)
            loss = loss_fn_dict[loss_fn](ind_pred, target)

            if self.MGDA:
                self.optimizer_encoder.zero_grad()
                self.optimizer_head.zero_grad()
                loss.backward()
                self.optimizer_encoder.step()
                self.optimizer_head.step()
            else:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss, result

    def infer_completion(self, data, batch_size):
        bev_seq = data['bev_seq']
        trans_matrices = data['trans_matrices']
        num_agent = data['num_agent']

        result, ind_pred = self.model(bev_seq, trans_matrices, num_agent, batch_size=batch_size)

        return result

    # Used my MAE model in scene completion, added by Juexiao
    def step_mae_completion(self, data, batch_size, mask_ratio, loss_fn='mse', trainable = False):
        # figure out the dimensions: squeeze, permute etc
        bev_seq = data['bev_seq'].squeeze(1).permute(0,3,1,2)
        bev_seq_next = data['bev_seq_next'] # [bxa, ts-1, H, W, C]
        if bev_seq_next.dim()>2:
            bev_seq_next = bev_seq_next.permute(0,1,4,2,3) # [bxa, ts-1, C, H, W]
        # print(data['bev_seq_teacher'].size())
        bev_teacher = data['bev_seq_teacher'].squeeze(1).permute(0,3,1,2)
        # print("teacher size", bev_teacher.size())
        num_agent_tensor = data['num_agent']
        trans_matrices = data['trans_matrices']
        
        with torch.cuda.amp.autocast(enabled=False):
            loss, result, _, ind_pred = self.model(bev_seq, 
                                                    bev_seq_next, 
                                                    bev_teacher, 
                                                    trans_matrices, 
                                                    num_agent_tensor,
                                                    batch_size,
                                                    mask_ratio=mask_ratio)
        # print("result size", result.size())
        # exit(1)
        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            # torch.save(bev_seq.cpu(), "lossnan-bev.pt")
            # torch.save(self.model.cpu(), "lossnan-mae.pt")
            # torch.save(result.cpu(), "lossnan-pred.pt")
            # torch.save(mask.cpu(), "lossnan-mask.pt")
            sys.exit(1)

        if trainable:
            self.mae_loss_scaler(loss, self.optimizer, parameters=self.model.parameters(),
                        update_grad=True)
            self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()

        return loss_value, result, ind_pred.detach()

    # inference mae, ego use groundtruth
    def infer_mae_completion(self, data, batch_size, mask_ratio, loss_fn='mse'):
        # figure out the dimensions: squeeze, permute etc
        bev_seq = data['bev_seq'].squeeze(1).permute(0,3,1,2)
        bev_seq_next = data['bev_seq_next'] # [bxa, ts-1, H, W, C]
        if bev_seq_next.dim()>2:
            bev_seq_next = bev_seq_next.permute(0,1,4,2,3) # [bxa, ts-1, C, H, W]
        # print(data['bev_seq_teacher'].size())
        bev_teacher = data['bev_seq_teacher'].squeeze(1).permute(0,3,1,2)
        # print("teacher size", bev_teacher.size())
        num_agent_tensor = data['num_agent']
        trans_matrices = data['trans_matrices']
        
        with torch.cuda.amp.autocast(enabled=False):
            loss, result, latent, ind_pred = self.model.inference(bev_seq, 
                                                    bev_seq_next, 
                                                    bev_teacher, 
                                                    trans_matrices, 
                                                    num_agent_tensor,
                                                    batch_size,
                                                    mask_ratio=mask_ratio)
        
        loss_value = loss.item()
        return loss_value, result, ind_pred.detach(), latent.detach()

    def step_vae_completion(self, data, batch_size, loss_fn='ce', trainable=False):
        bev_seq = data['bev_seq']
        trans_matrices = data['trans_matrices']
        num_agent = data['num_agent']

        vq_loss, result, ind_recon, perplexity = self.model(bev_seq, trans_matrices, num_agent, batch_size=batch_size)
        target = bev_seq.squeeze(1).permute(0,3,1,2).detach()
        # print("target", target.size())
        # recon_error = F.mse_loss(ind_recon, target)
        ## classification task ##
        # target = bev_seq.permute(0, 1, 4, 2, 3).squeeze(1)
        target = target.type(torch.LongTensor).to(ind_recon.device)
        # print("target", target.size())
        # print("ind_recon", ind_recon.size(), ind_recon.type())
        loss_fn = nn.CrossEntropyLoss()
        recon_error = loss_fn(ind_recon, target)
        loss = recon_error + vq_loss

        if trainable:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss.item(), result, ind_recon, perplexity.detach()

    def infer_vae_completion(self, data, batch_size, loss_fn='ce', trainable=False):
        bev_seq = data['bev_seq']
        trans_matrices = data['trans_matrices']
        num_agent = data['num_agent']

        vq_loss, result, ind_recon = self.model(bev_seq, trans_matrices, num_agent, batch_size=batch_size)

        return result

    # Used by VQ STAR model in scene completion, added by Juexiao
    def step_vqstar_completion(self, data, batch_size, mask_ratio, loss_fn='mse', trainable = False):
        # figure out the dimensions: squeeze, permute etc
        bev_seq = data['bev_seq'].squeeze(1).permute(0,3,1,2)
        bev_seq_next = data['bev_seq_next'] # [bxa, ts-1, H, W, C]
        if bev_seq_next.dim()>2:
            bev_seq_next = bev_seq_next.permute(0,1,4,2,3) # [bxa, ts-1, C, H, W]
        # print(data['bev_seq_teacher'].size())
        bev_teacher = data['bev_seq_teacher'].squeeze(1).permute(0,3,1,2)
        # print("teacher size", bev_teacher.size())
        num_agent_tensor = data['num_agent']
        trans_matrices = data['trans_matrices']
        
        with torch.cuda.amp.autocast(enabled=False):
            loss, result, _, ind_pred, perplexity = self.model(bev_seq, 
                                                    bev_seq_next, 
                                                    bev_teacher, 
                                                    trans_matrices, 
                                                    num_agent_tensor,
                                                    batch_size,
                                                    mask_ratio=mask_ratio)
        # print("result size", result.size())
        # exit(1)
        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if trainable:
            self.mae_loss_scaler(loss, self.optimizer, parameters=self.model.parameters(),
                        update_grad=True)
            self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()

        return loss_value, result, ind_pred.detach(), perplexity.detach()

    def infer_vqstar_completion(self, data, batch_size, mask_ratio, loss_fn='mse'):
        # figure out the dimensions: squeeze, permute etc
        bev_seq = data['bev_seq'].squeeze(1).permute(0,3,1,2)
        bev_seq_next = data['bev_seq_next'] # [bxa, ts-1, H, W, C]
        if bev_seq_next.dim()>2:
            bev_seq_next = bev_seq_next.permute(0,1,4,2,3) # [bxa, ts-1, C, H, W]
        # print(data['bev_seq_teacher'].size())
        bev_teacher = data['bev_seq_teacher'].squeeze(1).permute(0,3,1,2)
        # print("teacher size", bev_teacher.size())
        num_agent_tensor = data['num_agent']
        trans_matrices = data['trans_matrices']
        
        with torch.cuda.amp.autocast(enabled=False):
            loss, result, ind_pred, perplexity, encodings = self.model.inference(bev_seq, 
                                                    bev_seq_next, 
                                                    bev_teacher, 
                                                    trans_matrices, 
                                                    num_agent_tensor,
                                                    batch_size,
                                                    mask_ratio=mask_ratio)
        
        loss_value = loss.item()
        return loss_value, result, ind_pred.detach(), perplexity.detach(), encodings.detach()

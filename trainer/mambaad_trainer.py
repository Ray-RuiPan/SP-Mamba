import os
import copy
import glob
import shutil
import datetime
import time
import torch.distributed as dist
import tabulate
import torch
import torch.nn as nn


from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term
from util.net import trans_state_dict, print_networks, get_timepc, reduce_tensor
from util.net import get_loss_scaler, get_autocast, distribute_bn
from optim.scheduler import get_scheduler
from data import get_loader
from model import get_model
from optim import get_optim
from loss import get_loss_terms
from util.metric import get_evaluator, Evaluator
from timm.data import Mixup

import numpy as np
from torch.nn.parallel import DistributedDataParallel as NativeDDP

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model as ApexSyncBN
except:
    from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN
from timm.layers.norm_act import convert_sync_batchnorm as TIMMSyncBN
from timm.utils import dispatch_clip_grad

from ._base_trainer import BaseTrainer
from . import TRAINER
from util.vis import vis_rgb_gt_amp


@TRAINER.register_module
class MAMBAADTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(MAMBAADTrainer, self).__init__(cfg)
        #self.distance = nn.Parameter(self.net.net_s.prototype_distance256)
        #self.loss_pdistance = nn.Parameter(torch.mean(self.distance))

    def set_input(self, inputs):
        self.imgs = inputs['img'].cuda()
        self. imgs_mask = inputs['img_mask'].cuda()
        self.cls_name = inputs['cls_name']
        self.anomaly = inputs['anomaly']
        self.img_path = inputs['img_path']
        self.bs = self.imgs.shape[0]


    def forward(self):
        self.feats_t, self.feats_s = self.net(self.imgs)


    def optimize_parameters(self):
        if self.mixup_fn is not None:
            self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
        with self.amp_autocast():
            self.forward()
            loss_mse = self.loss_terms['pixel'](self.feats_t, self.feats_s)
            # loss_pdistance = self.loss_terms['pdistance'](self.feats_t, self.feats_s)


        prototype_distance256 = self.net.net_s.prototype_distance256
        prototype_distance256 = torch.mean(prototype_distance256)
        loss_pdistance = prototype_distance256
        latent_loss_weight = 25
        loss = loss_mse + latent_loss_weight * loss_pdistance



        self.backward_term(loss, self.optim)

        update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss_mse, self.world_size).clone().detach().item(),
                        1,
                      self.master)

        
        update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss_pdistance, self.world_size).clone().detach().item(),
                        1,
                        self.master)


        update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss, self.world_size).clone().detach().item(),
                        1,
                        self.master)



    # def train(self):
    #     self.reset(isTrain=True)
    #     self.train_loader.sampler.set_epoch(int(self.epoch)) if self.cfg.dist else None
    #     train_length = self.cfg.data.train_size
    #     train_loader = iter(self.train_loader)
    #     while self.epoch < self.epoch_full and self.iter < self.iter_full:
    #         self.scheduler_step(self.iter)
    #         # ---------- data ----------
    #         t1 = get_timepc()
    #         self.iter += 1
    #         train_data = next(train_loader)
    #         self.set_input(train_data)
    #         t2 = get_timepc()
    #         update_log_term(self.log_terms.get('data_t'), t2 - t1, 1, self.master)
    #         # ---------- optimization ----------
    #         self.optimize_parameters()
    #         t3 = get_timepc()
    #         update_log_term(self.log_terms.get('optim_t'), t3 - t2, 1, self.master)
    #         update_log_term(self.log_terms.get('batch_t'), t3 - t1, 1, self.master)
    #         self.cfg.total_time = get_timepc() - self.cfg.task_start_time
    #         self.save_checkpoint()
    #         break
    #         # ---------- log ----------
    #         if self.master:
    #             if self.iter % self.cfg.logging.train_log_per == 0:
    #                 msg = able(self.progress.get_msg(self.iter, self.iter_full, self.iter / train_length,
    #                                                  self.iter_full / train_length), self.master, None)
    #                 log_msg(self.logger, msg)
    #                 if self.writer:
    #                     for k, v in self.log_terms.items():
    #                         self.writer.add_scalar(f'Train/{k}', v.val, self.iter)
    #                     self.writer.flush()
    #         if self.iter % self.cfg.logging.train_reset_log_per == 0:
    #             self.reset(isTrain=True)
    #         # ---------- update train_loader ----------
    #         if self.iter % train_length == 0:
    #             self.epoch += 1
    #             if self.cfg.dist and self.dist_BN != '':
    #                 distribute_bn(self.net, self.world_size, self.dist_BN)
    #             self.optim.sync_lookahead() if hasattr(self.optim, 'sync_lookahead') else None
    #             if self.epoch >= self.cfg.trainer.test_start_epoch or self.epoch % self.cfg.trainer.test_per_epoch == 0:
    #                 self.test()
    #             else:
    #                 self.test_ghost()
    #             self.cfg.total_time = get_timepc() - self.cfg.task_start_time
    #             total_time_str = str(datetime.timedelta(seconds=int(self.cfg.total_time)))
    #             eta_time_str = str(
    #                 datetime.timedelta(seconds=int(self.cfg.total_time / self.epoch * (self.epoch_full - self.epoch))))
    #             log_msg(self.logger,
    #                     f'==> Total time: {total_time_str}\t Eta: {eta_time_str} \tLogged in \'{self.cfg.logdir}\'')
    #             self.save_checkpoint()
    #             self.reset(isTrain=True)
    #             self.train_loader.sampler.set_epoch(int(self.epoch)) if self.cfg.dist else None
    #             train_loader = iter(self.train_loader)
    #     self._finish()

    @torch.no_grad()
    def test(self):
        if self.master:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.makedirs(self.tmp_dir, exist_ok=True)
        self.reset(isTrain=False)
        imgs_masks, anomaly_maps, cls_names, anomalys, sample_anomalys, sample_predicts = [], [], [], [], [], []
        batch_idx = 0
        test_length = self.cfg.data.test_size
        test_loader = iter(self.test_loader)
        while batch_idx < test_length:
            # if batch_idx == 10:
            # 	break
            t1 = get_timepc()
            batch_idx += 1
            test_data = next(test_loader)
            self.set_input(test_data)
            self.forward()

            prototype_distance256 = self.net.net_s.prototype_distance256
            loss_pdistance = torch.mean(prototype_distance256)
            latent_loss_weight = 25 # Different datasets generally correspond to different latent_loss_weights, which should be adjusted according to the actual effect.

            loss_mse = self.loss_terms['pixel'](self.feats_t, self.feats_s)
            loss = loss_mse + loss_pdistance * latent_loss_weight


            update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss_mse, self.world_size).clone().detach().item(),
                            1, self.master)
            update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss_pdistance, self.world_size).clone().detach().item(),
                            1, self.master)


            update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss, self.world_size).clone().detach().item(),
                            1, self.master)

            # get anomaly maps
            anomaly_map, _ = self.evaluator.cal_anomaly_map(self.feats_t, self.feats_s,
                                                            [self.imgs.shape[2], self.imgs.shape[3]], uni_am=False,
                                                            amap_mode='add', gaussian_sigma=4)

            prototype_distance256 = prototype_distance256.cpu()
            prototype_distance256 = prototype_distance256.numpy()
            original_anomaly_map = anomaly_map


            alpha = 0.5 # Different datasets generally correspond to different alphas, which should be adjusted according to the actual effect.
            anomaly_map = anomaly_map + alpha * prototype_distance256 # anomaly_map:[16, 256, 256]
            prototype_distance256 = alpha * prototype_distance256

            # self.imgs_mask[self.imgs_mask > 0.], self.imgs_mask[self.imgs_mask <= 0.] = 1, 0
            self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
            if self.cfg.vis:
                if self.cfg.vis_dir is not None:
                    root_out = self.cfg.vis_dir
                else:
                    root_out = self.writer.logdir
                vis_rgb_gt_amp(self.img_path, self.imgs, self.imgs_mask.cpu().numpy().astype(int), anomaly_map, prototype_distance256, original_anomaly_map, self.feats_s,
                               self.cfg.model.name, root_out, self.cfg.data.root.split('/')[1])
            imgs_masks.append(self.imgs_mask.cpu().numpy().astype(int))
            anomaly_maps.append(anomaly_map)
            cls_names.append(np.array(self.cls_name))
            anomalys.append(self.anomaly.cpu().numpy().astype(int))
            t2 = get_timepc()
            update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)
            print(f'\r{batch_idx}/{test_length}', end='') if self.master else None
            # ---------- log ----------
            if self.master:
                if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
                    msg = able(self.progress.get_msg(batch_idx, test_length, 0, 0, prefix=f'Test'), self.master, None)
                    log_msg(self.logger, msg)
        # merge results
        if self.cfg.dist:
            results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
            torch.save(results, f'{self.tmp_dir}/{self.rank}.pth', _use_new_zipfile_serialization=False)
            if self.master:
                results = dict(imgs_masks=[], anomaly_maps=[], cls_names=[], anomalys=[])
                valid_results = False
                while not valid_results:
                    results_files = glob.glob(f'{self.tmp_dir}/*.pth')
                    if len(results_files) != self.cfg.world_size:
                        time.sleep(1)
                    else:
                        idx_result = 0
                        while idx_result < self.cfg.world_size:
                            results_file = results_files[idx_result]
                            try:
                                result = torch.load(results_file)
                                for k, v in result.items():
                                    results[k].extend(v)
                                idx_result += 1
                            except:
                                time.sleep(1)
                        valid_results = True
        else:
            results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
        if self.master:
            results = {k: np.concatenate(v, axis=0) for k, v in results.items()}
            msg = {}
            for idx, cls_name in enumerate(self.cls_names):
                metric_results = self.evaluator.run(results, cls_name, self.logger)
                msg['Name'] = msg.get('Name', [])
                msg['Name'].append(cls_name)
                avg_act = True if len(self.cls_names) > 1 and idx == len(self.cls_names) - 1 else False
                msg['Name'].append('Avg') if avg_act else None
                # msg += f'\n{cls_name:<10}'
                for metric in self.metrics:
                    metric_result = metric_results[metric] * 100
                    self.metric_recorder[f'{metric}_{cls_name}'].append(metric_result)
                    max_metric = max(self.metric_recorder[f'{metric}_{cls_name}'])
                    max_metric_idx = self.metric_recorder[f'{metric}_{cls_name}'].index(max_metric) + 1
                    msg[metric] = msg.get(metric, [])
                    msg[metric].append(metric_result)
                    msg[f'{metric} (Max)'] = msg.get(f'{metric} (Max)', [])
                    msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
                    if avg_act:
                        metric_result_avg = sum(msg[metric]) / len(msg[metric])
                        self.metric_recorder[f'{metric}_Avg'].append(metric_result_avg)
                        max_metric = max(self.metric_recorder[f'{metric}_Avg'])
                        max_metric_idx = self.metric_recorder[f'{metric}_Avg'].index(max_metric) + 1
                        msg[metric].append(metric_result_avg)
                        msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
            msg = tabulate.tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.3f', numalign="center",
                                    stralign="center", )
            log_msg(self.logger, f'\n{msg}')



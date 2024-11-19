import logging
import time
import os
import torch

class unsupervise_Trainer(object):
    def __init__(self, cfg, model, rank, output_dir,logger=None,lr_scheduler=None):
        self.model = model
        self.output_dir = output_dir
        self.rank = rank
        self.print_freq = cfg.PRINT_FREQ
        self.vis=cfg.VIS
        self.logger=logger
        self.meter={}
        self.lr_scheduler=lr_scheduler

    def train(self, epoch, real_data_loader, camera_data_loader, optimizer):
        for meter in self.meter:
            self.meter[meter].reset()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        optimizer.zero_grad()
        self.model.train()
        if self.rank == 0:
            lr_msg='lr: {0}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
            self.logger.info(lr_msg)
        end = time.time()
        
        data_length = len(real_data_loader)

        for i,(real_data,camera_data) in enumerate(zip(real_data_loader, camera_data_loader)):
            batched_inputs = dict(
                syn = camera_data,
                real = real_data,
            )
            data_time.update(time.time() - end)

            loss_dict = self.model(batched_inputs)
            loss = 0
            num_images = real_data['points'].shape[0] + camera_data['points'].shape[0]

            for name in loss_dict:
                l=loss_dict[name]
                loss=loss+l
                if name not in self.meter:
                    self.meter[name]=AverageMeter(name)
                    self.meter[name].reset()
                self.meter[name].update(l.item(),num_images)
            
            
            if not self.vis:
                loss.backward()
                
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4, norm_type=2)
                optimizer.step()
                optimizer.zero_grad()
                
            batch_time.update(time.time() - end)
            end = time.time()
            

            if i % self.print_freq == 0 and self.rank == 0 :
                msg = 'Epoch: [{0}][{1}/{2}] ' \
                        'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) ' \
                        'Speed: {speed:.1f} samples/s ' \
                        'Data: {data_time_val:.3f}ms ({data_time_avg:.3f}ms)'.format(
                            epoch, i, data_length,
                            batch_time=batch_time,
                            speed=num_images / batch_time.val,
                            data_time_val=(data_time.val)*1000,
                            data_time_avg=(data_time.avg)*1000
                        )
                for name in self.meter:
                    msg+='{l}'.format(
                            l=_get_loss_info(self.meter[name],name)
                        )
                self.logger.info(msg)

            if self.lr_scheduler:
                self.lr_scheduler.step()
            if i==data_length-1:
                break


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self,name=None):
        self.reset()
        self.name=name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

class Trainer(object):
    def __init__(self, cfg, model, rank, output_dir,logger=None,lr_scheduler=None):
        self.model = model
        self.output_dir = output_dir
        self.rank = rank
        self.print_freq = cfg.PRINT_FREQ
        self.vis=cfg.VIS
        self.logger=logger
        self.loss_name=cfg.MODEL.loss_name
        self.meter={name:AverageMeter(name) for name in self.loss_name}
        self.lr_scheduler=lr_scheduler

    def train(self, epoch, data_loader, optimizer):
        for meter in self.meter:
            self.meter[meter].reset()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        optimizer.zero_grad()
        self.model.train()
        if self.rank == 0:
            lr_msg='lr: {0}'.format(self.lr_scheduler.get_last_lr()[0])
            self.logger.info(lr_msg)
        end = time.time()
            
        for i,batched_inputs in enumerate(data_loader):

            data_time.update(time.time() - end)

            loss_dict = self.model(batched_inputs)
            loss = 0
            num_images = (batched_inputs['points'].shape[0])

            for name in loss_dict:
                l=loss_dict[name]
                loss=loss+l
                self.meter[name].update(l.item(),num_images)
            
            
            if not self.vis:
                loss.backward()
                
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4, norm_type=2)
                optimizer.step()
                optimizer.zero_grad()
                
            batch_time.update(time.time() - end)
            end = time.time()
            

            if i % self.print_freq == 0 and self.rank == 0 :
                msg = 'Epoch: [{0}][{1}/{2}] ' \
                        'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) ' \
                        'Speed: {speed:.1f} samples/s ' \
                        'Data: {data_time_val:.3f}ms ({data_time_avg:.3f}ms)'.format(
                            epoch, i, len(data_loader),
                            batch_time=batch_time,
                            speed=num_images / batch_time.val,
                            data_time_val=(data_time.val)*1000,
                            data_time_avg=(data_time.avg)*1000
                        )
                for name in self.meter:
                    msg+='{l}'.format(
                            l=_get_loss_info(self.meter[name],name)
                        )
                self.logger.info(msg)

            if self.lr_scheduler:
                self.lr_scheduler.step()





def _get_loss_info(meter, loss_name):
    msg = '{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(name=loss_name, meter=meter)
    return msg


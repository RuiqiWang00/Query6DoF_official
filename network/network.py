import torch
import torch.nn as nn
import os
from .decoder import DECODER_REGISTRY
from .encoder import ENCODER_REGISTRY
from .loss import LOSS_REGISTRY, chamfer_lossv2, r_lossv2, t_loss, s_loss
from mmengine import Registry
from collections import OrderedDict
NETWORK_REGISTRY = Registry("NETWORK")


@NETWORK_REGISTRY.register_module()
class Sparsenetv7(nn.Module):
    def __init__(self,
                backbone=dict(
                    type='PointNetfeat'
                ),
                decoder=dict(
                    type='sparse_decoder'
                ),
                pose_estimate=dict(
                    type='pose_estimater'
                ),
                name='Posenet',
                training=False,
                input_dim=256,
                n_pts=128,
                cat_num=6,
                losses=[],
                loss_name=[],
                vis=False
                ) -> None:
        super().__init__()
        self.backbone=ENCODER_REGISTRY.build(backbone)
        self.pose_estimater=DECODER_REGISTRY.build(pose_estimate)
        self.decoder=DECODER_REGISTRY.build(decoder)
        prior_feat=(2*torch.rand((cat_num,n_pts,input_dim))-1)/input_dim
        self.prior_feat=torch.nn.parameter.Parameter(data=prior_feat,requires_grad=True)
        self.cat_num=cat_num
        self.training=training
        self.sym_id=torch.Tensor([0,1,3])
        self.count=0
        if training:
            self.losses=[LOSS_REGISTRY.build(loss) for loss in losses]
            self.loss_name=loss_name
        self.vis=vis
        self.time_series=False

    def forward(self,batched_inputs, prior_feat_ = None, semi_supervise=False):
        points=batched_inputs['points']
        category=batched_inputs['cat_id'] #B
        if self.training:
            nocs=batched_inputs['nocs']
            model=batched_inputs['model']
            R=batched_inputs['R']
            s=batched_inputs['s']
            gt_green=batched_inputs['gt_green']
            gt_red=batched_inputs['gt_red']
            mean_shape=batched_inputs['mean_shape']
            t=batched_inputs['t']
            s_delta=batched_inputs['dimension_delta']
        mean_t=points.mean(dim=1)
        encoder_input=points-mean_t.unsqueeze(dim=1)
        encoder_out=self.backbone(encoder_input)
        inst_feat=encoder_out.transpose(1,2).contiguous()
        prior_feat=self.prior_feat[category,...] if prior_feat_ is None else prior_feat_

        device=torch.cuda.current_device()
        index=category+torch.arange(encoder_input.shape[0],dtype=torch.long,device=device)*self.cat_num

        sym=batched_inputs['sym']

        if self.training:
            if self.time_series:
                inst_feat,coord,response_coord,prior_feat=self.decoder(prior_feat,inst_feat,index,encoder_input)
            else:
                inst_feat,coord,response_coord=self.decoder(prior_feat,inst_feat,index,encoder_input)
            pred_r,pred_t,pred_s=self.pose_estimater(inst_feat,index)
            pred_t=pred_t+mean_t
            if semi_supervise:
                return pred_r.detach(), pred_t.detach(), pred_s.detach()
            loss_dict=self.train_forward(pred_r,pred_t,pred_s,gt_green,gt_red,t,s_delta,sym,coord,model,nocs,response_coord,prior_feat,points,R,s,mean_shape)
            return loss_dict

        else:
            mean_shape=batched_inputs['mean_shape']
            if not self.vis:
                if self.time_series:
                    inst_feat,prior_feat =self.decoder(prior_feat,inst_feat,index,encoder_input)
                else:
                    inst_feat=self.decoder(prior_feat,inst_feat,index,encoder_input)
            else:
                inst_feat,coord,response_coord,iam1,iam2=self.decoder(prior_feat,inst_feat,index,encoder_input)
                B,N=iam1.shape[0],iam1.shape[1]
                iam1=iam1.view(B,N,-1,4)
                M=iam2.shape[1]
                iam2=iam2.view(B,M,-1,4)
            pred_r,pred_t,pred_s=self.pose_estimater(inst_feat,index)
            pred_t=pred_t+mean_t
            if semi_supervise:
                return pred_r.detach(), pred_t.detach(), pred_s.detach()
            #pred_r:B,3,3 pred_t:B,3 pred_s:B,3
            pred_s=pred_s+mean_shape
            B=pred_r.shape[0]
            trans=torch.zeros((B,4,4),device=device)
            nocs_scale=torch.linalg.norm(pred_s,dim=-1,keepdim=True) #B,1
            trans[:,3,3]=1

            theta_x_=pred_r[:,0,0]-pred_r[:,2,2]#B
            theta_y_=pred_r[:,0,2]-pred_r[:,2,0]#B
            r_norm_=(theta_x_**2+theta_y_**2)**0.5 #B
            theta_x_=theta_x_/r_norm_
            theta_y_=theta_y_/r_norm_
            s_map_=torch.zeros((B,3,3),device=device) #B,3,3
            s_map_[:,1,1]=1
            s_map_[:,0,0],s_map_[:,0,2],s_map_[:,2,0],s_map_[:,2,2]=theta_x_,-theta_y_,theta_y_,theta_x_
            delta_r=torch.bmm(pred_r,s_map_) #B,3,3

            mask=torch.isin(category,self.sym_id.to(device))
            pred_r[mask,...]=delta_r[mask,...]

            trans[:,:3,:3]=pred_r*(nocs_scale.unsqueeze(dim=-1))
            trans[:,:3,3:]=pred_t.unsqueeze(dim=-1)
            size=pred_s/nocs_scale
            trans=trans.cpu().numpy()
            size=size.cpu().numpy()
            if not self.vis:
                if self.time_series:
                    return trans,size,prior_feat
                return trans,size
            else:
                return trans,size,coord.cpu(),response_coord.cpu(),iam1.cpu(),iam2.cpu(),pred_r.cpu().numpy(),pred_t.cpu().numpy(),nocs_scale.cpu().numpy()


    def train_forward(self,pred_r,pred_t,pred_s,gt_green,gt_red,t,s_delta,sym,coord,model,nocs,response_coord,prior_feat,points,R,s,mean_shape):
        '''
        
        '''
        paras={
            'chamfer':(coord,model),
            'r':(pred_r,gt_red,gt_green,sym),
            't':(pred_t,t),
            's':(pred_s,s_delta),
            'nocs':(response_coord,nocs),
            'consistency':(points,response_coord,pred_r,pred_t,torch.linalg.norm(pred_s+mean_shape,dim=-1,keepdim=True).unsqueeze(dim=-1)),
        }
        return {name:loss(*(paras[name])) for loss,name in zip(self.losses,self.loss_name)}
    

    # def unsupervise_train_forward(self,coord,response_coord):
@NETWORK_REGISTRY.register_module()
class unsupervise_model(Sparsenetv7):
    def __init__(self,
                backbone=dict(
                    type='PointNetfeat'
                ),
                decoder=dict(
                    type='sparse_decoder'
                ),
                pose_estimate=dict(
                    type='pose_estimater'
                ),
                name='Posenet',
                training=False,
                input_dim=256,
                n_pts=128,
                cat_num=6,
                losses=[],
                loss_name=[],
                vis=False,
                unsupervised=False,
                pose_loss_weight = 1.0,
                chamfer_loss_weight = 1.0,
                align_loss_weight=0.02
                ) -> None:
        super().__init__(backbone=backbone,
                decoder=decoder,
                pose_estimate=pose_estimate,
                name=name,
                training=training,
                input_dim=input_dim,
                n_pts=n_pts,
                cat_num=cat_num,
                losses=losses,
                loss_name=loss_name,
                vis=vis)
        self.unsupervised=unsupervised
        self.coord_loss=chamfer_lossv2(weight=chamfer_loss_weight)
        self.pose_loss_weight = pose_loss_weight
        self.r_loss = r_lossv2(weight=pose_loss_weight, beta=0.001)
        self.t_loss = t_loss(weight=pose_loss_weight,beta=0.005)
        self.s_loss = s_loss(weight=pose_loss_weight, beta=0.005)
        self.align_loss_weight = align_loss_weight

    def forward(self,batched_inputs,prior_feat_=None):
        if not (self.training and self.unsupervised):
            return super().forward(batched_inputs,prior_feat_)
        
        syn_data = batched_inputs['syn']
        real_data = batched_inputs['real']
        
        syn_loss_dict = super().forward(syn_data)
        real_loss_dict = self.unsupervise_train(real_data)
        # b1=syn_data['points'].shape[0]
        # b2=real_data['points'].shape[0]
        # for key in syn_loss_dict:
        #     syn_loss_dict[key]=syn_loss_dict[key]*b1/(b1+b2)
        # for key in real_loss_dict:
        #     real_loss_dict[key] = real_loss_dict *b2/(b1+b2)
        syn_loss_dict.update(real_loss_dict)
        return syn_loss_dict

    def unsupervise_train(self,batched_inputs):
        points=batched_inputs['points']
        category=batched_inputs['cat_id']
        mean_shape=batched_inputs['mean_shape']
        sym=batched_inputs['sym']
        pred1 = self.delta_unsupervise(points,mean_shape,category)
        pred2 = self.delta_unsupervise(points,mean_shape,category)
        return self.unsupervise_loss(pred1,pred2,sym,points)

    def unsupervise_loss(self,pred1, pred2,sym,points):
        r1=pred1['pred_r']
        t1=pred1['pred_t']
        s1=pred1['pred_s']
        r2=pred2['pred_r']
        t2=pred2['pred_t']
        s2=pred2['pred_s']
        # align_loss=self.align_loss(pred1,pred2,points)
        r_loss = self.r_loss(r1,r2[...,0:1],r2[...,1:2],sym)
        t_loss=self.t_loss(t1,t2)
        s_loss=self.s_loss(s1,s2)
        # pose_loss = self.PoseDis(r1,t1,s1,r2,t2,s2)
        loss_chamfer = self.coord_loss(pred1['coord'],pred2['coord'])
        return dict(
            un_pose_loss = r_loss+t_loss+s_loss,
            un_chamfer = loss_chamfer,
            # align_loss=align_loss
        )
    
    def align_loss(self,pred1,pred2,points):
        r1=pred1['pred_r']
        t1=pred1['pred_t_gt']
        s1=pred1['pred_s_gt']
        r2=pred2['pred_r']
        t2=pred2['pred_t_gt']
        s2=pred2['pred_s_gt']
        def get_gt(R,t,s,points):
            s=torch.linalg.norm(s,dim=-1,keepdim=True)
            return torch.bmm(R.transpose(1,2)/s[...,None],(points-t.unsqueeze(dim=1)).transpose(1,2)).transpose(1,2)
        return self.align_loss_weight*(torch.nn.functional.smooth_l1_loss(pred1['response_coord'],get_gt(r1,t1,s1,points),beta=0.1,reduction='none').sum(-1).mean()+\
            torch.nn.functional.smooth_l1_loss(pred2['response_coord'],get_gt(r2,t2,s2,points),beta=0.1,reduction='none').sum(-1).mean())
    

    def PoseDis(self,r1, t1, s1, r2, t2, s2):
        '''
        r1, r2: b*3*3
        t1, t2: b*3
        s1, s2: b*3
        '''
        dis_r = torch.mean(torch.norm(r1 - r2, dim=1))
        dis_t = torch.mean(torch.norm(t1 - t2, dim=1))
        dis_s = torch.mean(torch.norm(s1 - s2, dim=1))

        return self.pose_loss_weight *(dis_r + dis_t + dis_s)
        

    def delta_unsupervise(self, points, mean_shape, category):
        mean_t=points.mean(dim=1)
        encoder_input=points-mean_t.unsqueeze(dim=1)

        b=points.shape[0]
        delta_r1 = torch.rand(b, 6).cuda()
        delta_r1 = self.Ortho6d2Mat(delta_r1[:, :3].contiguous(), delta_r1[:, 3:].contiguous()).view(-1,3,3)
        delta_s1 = torch.rand(b, 1).cuda()
        delta_s1 = delta_s1.uniform_(0.8, 1.2)
        pts1 = (encoder_input) / delta_s1.unsqueeze(2) @ delta_r1

        encoder_out=self.backbone(pts1)
        inst_feat=encoder_out.transpose(1,2).contiguous()
        prior_feat=self.prior_feat[category,...]

        device=torch.cuda.current_device()
        index=category+torch.arange(encoder_input.shape[0],dtype=torch.long,device=device)*self.cat_num

        inst_feat,coord,response_coord=self.decoder(prior_feat,inst_feat,index,encoder_input)
        pred_r1,pred_t1,pred_s1=self.pose_estimater(inst_feat,index)

        pred_r1 = delta_r1 @ pred_r1
        pred_t1 = delta_s1[...,None] * torch.bmm(delta_r1, pred_t1[...,None])
        pred_t=pred_t1.squeeze(dim=-1)+mean_t
        pred_s1 = delta_s1 *(pred_s1+mean_shape) -mean_shape
        return dict(
            pred_r=pred_r1,
            pred_t=pred_t,
            pred_s=pred_s1,
            coord = coord,
            response_coord = response_coord,
            pred_t_gt=pred_t+mean_t,
            pred_s_gt=pred_s1+mean_shape
        )



    def Ortho6d2Mat(self,x_raw, y_raw):
        y = self.normalize_vector(y_raw)
        z = self.cross_product(x_raw, y) #B,3
        z = self.normalize_vector(z)#B,3
        x = self.cross_product(y,z)#B,3

        x = x.unsqueeze(2)
        y = y.unsqueeze(2)
        z = z.unsqueeze(2)
        matrix = torch.cat((x,y,z),dim=2) #batch*3*3
        return matrix
    
    def normalize_vector(self, v, dim =1, return_mag =False):
        return torch.nn.functional.normalize(v,dim=dim)

    def cross_product(self,u, v):
        return torch.cross(u,v,dim=-1)

@NETWORK_REGISTRY.register_module()
class mean_teacher(nn.Module):
    def __init__(self,
                model,
                ema_gamma):
        super().__init__()
        self.model = NETWORK_REGISTRY.build(model)
        self.ema_model = NETWORK_REGISTRY.build(model)
        for param in self.ema_model.parameters():
            param.requires_grad = False
        self.ema_gamma=ema_gamma
        self.load=False

    def init_model(self,init_path):
        checkpoint = torch.load(init_path, map_location=lambda storage, loc: storage)
        checkpoint = checkpoint['state_dict']
        if 'module' in (list(checkpoint.keys()))[0]:
            new_checkpoint = OrderedDict()
            for key in checkpoint.keys():
                new_checkpoint['.'.join(key.split('.')[1:])] = checkpoint[key]
            checkpoint = new_checkpoint
        self.model.load_state_dict(checkpoint)
        self.ema_model.load_state_dict(checkpoint)
        self.load=True
        

    def forward(self,batched_inputs):
        assert self.training and self.load
        syn_data = batched_inputs['syn']
        real_data = batched_inputs['real']
        with torch.no_grad():
            self.ema_model.eval()
            pred_r_teacher, pred_t_teacher, pred_s_teacher = self.ema_model(real_data, semi_supervise=True)
        real_data['R'] = pred_r_teacher
        real_data['s_delta'] = pred_s_teacher
        real_data['t'] = pred_t_teacher
        real_data['gt_green'] , real_data['gt_red'] = self.get_gt_v(pred_r_teacher)
        syn_loss_dict=self.model(syn_data)
        real_loss_dict=self.model(real_data)
        real_loss_dict = {"semi_" + key: value for key, value in real_loss_dict.items()}
        syn_loss_dict.update(real_loss_dict)
        return syn_loss_dict

    def ema_update(self):
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.ema_gamma).add_(1 - self.ema_gamma, param.data)

    def get_gt_v(self,Rs, axis=2):
        # TODO use 3 axis, the order remains: do we need to change order?
        if axis == 3:
            raise NotImplementedError
        else:
            assert axis == 2
            gt_green = Rs[:,:,1:2]
            gt_red = Rs[:,:,0:1]
        return gt_green, gt_red

@NETWORK_REGISTRY.register_module()
class time_series_model(nn.Module):
    def __init__(self, pose_model):
        super().__init__()
        self.pose_model = NETWORK_REGISTRY.build(pose_model)
        self.pose_model.decoder.time_series=True
        self.time_series=True

    def forward(self, batched_inputs):
        cur_frame = batched_inputs['cur_frame']
        another_frame = batched_inputs['another_frame']
        
        self.pose_model.eval()
        with torch.no_grad():
            trans, size, prior_feat = self.pose_model(another_frame)

            trans,size,prior_feat = self.pose_model(cur_frame, prior_feat)
        return trans, size

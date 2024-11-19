_base_=['_base_.py']
CFG_NAME = ''
OUTPUT_DIR = 'runs'
RUN_NAME = 'semi_supervise'
PRINT_FREQ = 25
DIST_BACKEND = 'nccl'
AUTO_RESUME = True
RESUME_FILE = ''
ONLY_MODEL = True
CHANGE_SCHEDULE = False
find_unused_parameters = False
VIS = False
DATA='real_test'
train=True

is_iter=True
AUTO_RESUME = True
RESUME_FILE = '/data/ruiqiwang/codes/Query6DoF_official/runs/Real/unsupervise/model/checkpoint_epoch_100.tar.pth'
MODEL = dict(
    type = 'mean_teacher',
    model = dict(        
        type='Sparsenetv7',
        name='Sparsenetv7',
        n_pts=64,
        backbone=dict(
            type='Pointnet2MSG',
            input_channels=0,
            mlp=[[256, 256], [256, 256], [256, 256], [512, 512]]),
        decoder=dict(
            type='deep_prior_decoderv2_9',
            group=4,
            input_dim=256,
            middle_dim=1024,
            training=train,
            cat_num=6),
        pose_estimate=dict(
            type='pose_estimater',
            input_dim=512,
            middle_dim=256
        ),
        input_dim=256,
        cat_num=6,
        training=train,
        loss_name=['r','t','s','chamfer','nocs'],
        losses=[
            dict(type='r_lossv2', weight=1.0,beta=0.001),
            dict(type='t_loss', weight=1,beta=0.005),
            dict(type='s_loss', weight=1.0,beta=0.005),
            dict(type='chamfer_lossv2',weight=3.0),
            dict(type='consistency_lossv2',weight=1.0,beta=0.1)
        ],),
    ema_gamma = 0.99
)


CAMERA_DATASET = dict(
    type='PoseDataset',
    source='CAMERA',
    mode='train',
    data_dir='data',
    n_pts=1024,
    vis=False,
    img_size=192,
    use_cache=False)

REAL_DATASET = dict(
    type='PoseDataset',
    source='Real',
    mode='train',
    data_dir='data',
    n_pts=1024,
    vis=False,
    img_size=192,
    use_cache=False)


CAMERA_DATALOADER = dict(
    type='DataLoader',
    batch_size=15,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    drop_last=True)

REAL_DATALOADER = dict(
    type='DataLoader',
    batch_size=15,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    drop_last=True)

# DATALOADER['persistent_workers']=DATALOADER['num_workers']>0
# if VIS:
#     DATALOADER['num_workers']=1


OPTIMIZER = dict(type='AdamW', lr=1e-4, weight_decay=1e-4)
SCHEDULER = dict(type='CosineAnnealingLR', T_max=4318*30//(15*4), eta_min=1e-7, last_epoch=-1, verbose=False)
TRAIN = dict(BEGIN_EPOCH=0, END_EPOCH=30, SAVE_EPOCH_STEP=10, VIS=False)
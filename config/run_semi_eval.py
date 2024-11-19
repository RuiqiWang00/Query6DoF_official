_base_=['_base_.py']
RUN_NAME = 'semi_supervise'
DATA='real_test'

train=False
MODEL = dict(
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
    ])
RESUME_FILE = 'checkpoint_epoch_10.tar.pth'
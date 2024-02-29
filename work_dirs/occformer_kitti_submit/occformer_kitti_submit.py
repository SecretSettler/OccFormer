point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
class_names = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
    'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
    'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
    'pole', 'traffic-sign'
]
dataset_type = 'CustomSemanticKITTILssDataset'
data_root = 'data/SemanticKITTI'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_SemanticKitti',
        is_train=True,
        data_config=dict(
            input_size=(384, 1280),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        img_norm_cfg=dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True)),
    dict(
        type='CreateDepthFromLiDAR',
        data_root='data/SemanticKITTI',
        dataset='kitti'),
    dict(
        type='LoadSemKittiAnnotation',
        bda_aug_conf=dict(
            rot_lim=(0, 0),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5,
            flip_dz_ratio=0.5),
        is_train=True,
        point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
    dict(
        type='OccDefaultFormatBundle3D',
        class_names=[
            'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck',
            'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 'road',
            'parking', 'sidewalk', 'other-ground', 'building', 'fence',
            'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
        ]),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_occ'],
        meta_keys=['pc_range', 'occ_size'])
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_SemanticKitti',
        is_train=False,
        data_config=dict(
            input_size=(384, 1280),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        img_norm_cfg=dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True)),
    dict(
        type='LoadSemKittiAnnotation',
        bda_aug_conf=dict(
            rot_lim=(0, 0),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5,
            flip_dz_ratio=0.5),
        is_train=False,
        point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
    dict(
        type='OccDefaultFormatBundle3D',
        class_names=[
            'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck',
            'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 'road',
            'parking', 'sidewalk', 'other-ground', 'building', 'fence',
            'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
        ],
        with_label=False),
    dict(
        type='Collect3D',
        keys=['img_inputs'],
        meta_keys=['pc_range', 'occ_size', 'sequence', 'frame_id'])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='CustomSemanticKITTILssDataset',
        data_root='data/SemanticKITTI',
        ann_file='data/SemanticKITTI/labels',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_SemanticKitti',
                is_train=True,
                data_config=dict(
                    input_size=(384, 1280),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                img_norm_cfg=dict(
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)),
            dict(
                type='CreateDepthFromLiDAR',
                data_root='data/SemanticKITTI',
                dataset='kitti'),
            dict(
                type='LoadSemKittiAnnotation',
                bda_aug_conf=dict(
                    rot_lim=(0, 0),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5,
                    flip_dz_ratio=0.5),
                is_train=True,
                point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
            dict(
                type='OccDefaultFormatBundle3D',
                class_names=[
                    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck',
                    'other-vehicle', 'person', 'bicyclist', 'motorcyclist',
                    'road', 'parking', 'sidewalk', 'other-ground', 'building',
                    'fence', 'vegetation', 'trunk', 'terrain', 'pole',
                    'traffic-sign'
                ]),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'gt_occ'],
                meta_keys=['pc_range', 'occ_size'])
        ],
        classes=[
            'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck',
            'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 'road',
            'parking', 'sidewalk', 'other-ground', 'building', 'fence',
            'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        split='train',
        camera_used=['left'],
        occ_size=[256, 256, 32],
        pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
    val=dict(
        type='CustomSemanticKITTILssDataset',
        ann_file='data/SemanticKITTI/labels',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_SemanticKitti',
                is_train=False,
                data_config=dict(
                    input_size=(384, 1280),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                img_norm_cfg=dict(
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)),
            dict(
                type='LoadSemKittiAnnotation',
                bda_aug_conf=dict(
                    rot_lim=(0, 0),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5,
                    flip_dz_ratio=0.5),
                is_train=False,
                point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
            dict(
                type='OccDefaultFormatBundle3D',
                class_names=[
                    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck',
                    'other-vehicle', 'person', 'bicyclist', 'motorcyclist',
                    'road', 'parking', 'sidewalk', 'other-ground', 'building',
                    'fence', 'vegetation', 'trunk', 'terrain', 'pole',
                    'traffic-sign'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs'],
                meta_keys=['pc_range', 'occ_size', 'sequence', 'frame_id'])
        ],
        classes=[
            'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck',
            'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 'road',
            'parking', 'sidewalk', 'other-ground', 'building', 'fence',
            'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        data_root='data/SemanticKITTI',
        split='test-submit',
        camera_used=['left'],
        occ_size=[256, 256, 32],
        pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
    test=dict(
        type='CustomSemanticKITTILssDataset',
        data_root='data/SemanticKITTI',
        ann_file='data/SemanticKITTI/labels',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_SemanticKitti',
                is_train=False,
                data_config=dict(
                    input_size=(384, 1280),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                img_norm_cfg=dict(
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)),
            dict(
                type='LoadSemKittiAnnotation',
                bda_aug_conf=dict(
                    rot_lim=(0, 0),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5,
                    flip_dz_ratio=0.5),
                is_train=False,
                point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
            dict(
                type='OccDefaultFormatBundle3D',
                class_names=[
                    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck',
                    'other-vehicle', 'person', 'bicyclist', 'motorcyclist',
                    'road', 'parking', 'sidewalk', 'other-ground', 'building',
                    'fence', 'vegetation', 'trunk', 'terrain', 'pole',
                    'traffic-sign'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs'],
                meta_keys=['pc_range', 'occ_size', 'sequence', 'frame_id'])
        ],
        classes=[
            'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck',
            'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 'road',
            'parking', 'sidewalk', 'other-ground', 'building', 'fence',
            'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        split='test-submit',
        camera_used=['left'],
        occ_size=[256, 256, 32],
        pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(
    interval=1,
    pipeline=[
        dict(
            type='LoadMultiViewImageFromFiles_SemanticKitti',
            is_train=False,
            data_config=dict(
                input_size=(384, 1280),
                resize=(-0.06, 0.11),
                rot=(-5.4, 5.4),
                flip=True,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            img_norm_cfg=dict(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True)),
        dict(
            type='LoadSemKittiAnnotation',
            bda_aug_conf=dict(
                rot_lim=(0, 0),
                scale_lim=(0.95, 1.05),
                flip_dx_ratio=0.5,
                flip_dy_ratio=0.5,
                flip_dz_ratio=0.5),
            is_train=False,
            point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
        dict(
            type='OccDefaultFormatBundle3D',
            class_names=[
                'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck',
                'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 'road',
                'parking', 'sidewalk', 'other-ground', 'building', 'fence',
                'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
            ],
            with_label=False),
        dict(
            type='Collect3D',
            keys=['img_inputs'],
            meta_keys=['pc_range', 'occ_size', 'sequence', 'frame_id'])
    ],
    save_best='semkitti_SSC_mIoU',
    rule='greater')
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=1,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/occformer_kitti_submit'
load_from = None
resume_from = None
workflow = [('train', 1)]
sync_bn = True
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
camera_used = ['left']
num_class = 20
occ_size = [256, 256, 32]
lss_downsample = [2, 2, 2]
voxel_x = 0.2
voxel_y = 0.2
voxel_z = 0.2
voxel_size = [0.2, 0.2, 0.2]
data_config = dict(
    input_size=(384, 1280),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.0)
grid_config = dict(
    xbound=[0, 51.2, 0.4],
    ybound=[-25.6, 25.6, 0.4],
    zbound=[-2, 4.4, 0.4],
    dbound=[2.0, 58.0, 0.5])
numC_Trans = 128
voxel_channels = [128, 256, 512, 1024]
voxel_num_layer = [2, 2, 2, 2]
voxel_strides = [1, 2, 2, 2]
voxel_out_indices = (0, 1, 2, 3)
voxel_out_channels = 192
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
mask2former_num_queries = 100
mask2former_feat_channel = 192
mask2former_output_channel = 192
mask2former_pos_channel = 64.0
mask2former_num_heads = 6
model = dict(
    type='OccupancyFormer',
    img_backbone=dict(
        type='CustomEfficientNet',
        arch='b7',
        drop_path_rate=0.2,
        frozen_stages=0,
        norm_eval=False,
        out_indices=(2, 3, 4, 5, 6),
        with_cp=True,
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone',
            checkpoint=
            'ckpts/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth'
        )),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[48, 80, 224, 640, 2560],
        upsample_strides=[0.25, 0.5, 1, 2, 2],
        out_channels=[128, 128, 128, 128, 128]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShootVoxel',
        numC_input=640,
        cam_channels=33,
        loss_depth_weight=1.0,
        grid_config=dict(
            xbound=[0, 51.2, 0.4],
            ybound=[-25.6, 25.6, 0.4],
            zbound=[-2, 4.4, 0.4],
            dbound=[2.0, 58.0, 0.5]),
        data_config=dict(
            input_size=(384, 1280),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        numC_Trans=128,
        vp_megvii=False),
    img_bev_encoder_backbone=dict(
        type='OccupancyEncoder',
        num_stage=4,
        in_channels=128,
        block_numbers=[2, 2, 2, 2],
        block_inplanes=[128, 256, 512, 1024],
        block_strides=[1, 2, 2, 2],
        out_indices=(0, 1, 2, 3),
        with_cp=True,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    img_bev_encoder_neck=dict(
        type='MSDeformAttnPixelDecoder3D',
        strides=[2, 4, 8, 16],
        in_channels=[128, 256, 512, 1024],
        feat_channels=192,
        out_channels=192,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        encoder=dict(
            type='DetrTransformerEncoder',
            num_layers=6,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention3D',
                    embed_dims=192,
                    num_heads=8,
                    num_levels=3,
                    num_points=4,
                    im2col_step=64,
                    dropout=0.0,
                    batch_first=False,
                    norm_cfg=None,
                    init_cfg=None),
                ffn_cfgs=dict(embed_dims=192),
                feedforward_channels=768,
                ffn_dropout=0.0,
                operation_order=('self_attn', 'norm', 'ffn', 'norm')),
            init_cfg=None),
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=64, normalize=True)),
    pts_bbox_head=dict(
        type='Mask2FormerOccHead',
        feat_channels=192,
        out_channels=192,
        num_queries=100,
        num_occupancy_classes=20,
        pooling_attn_mask=True,
        sample_weight_gamma=0.25,
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=64.0, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=192,
                    num_heads=6,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=192,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=1536,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1
            ]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        loss_metric=dict(type='MetricLoss'),
        point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
    train_cfg=dict(
        pts=dict(
            num_points=50176,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='MaskHungarianAssigner',
                cls_cost=dict(type='ClassificationCost', weight=2.0),
                mask_cost=dict(
                    type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dice_cost=dict(
                    type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
            sampler=dict(type='MaskPseudoSampler'))),
    test_cfg=dict(
        pts=dict(semantic_on=True, panoptic_on=False, instance_on=False)))
ann_file = 'data/SemanticKITTI/labels'
bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0.5)
test_config = dict(
    type='CustomSemanticKITTILssDataset',
    data_root='data/SemanticKITTI',
    ann_file='data/SemanticKITTI/labels',
    pipeline=[
        dict(
            type='LoadMultiViewImageFromFiles_SemanticKitti',
            is_train=False,
            data_config=dict(
                input_size=(384, 1280),
                resize=(-0.06, 0.11),
                rot=(-5.4, 5.4),
                flip=True,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            img_norm_cfg=dict(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True)),
        dict(
            type='LoadSemKittiAnnotation',
            bda_aug_conf=dict(
                rot_lim=(0, 0),
                scale_lim=(0.95, 1.05),
                flip_dx_ratio=0.5,
                flip_dy_ratio=0.5,
                flip_dz_ratio=0.5),
            is_train=False,
            point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
        dict(
            type='OccDefaultFormatBundle3D',
            class_names=[
                'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck',
                'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 'road',
                'parking', 'sidewalk', 'other-ground', 'building', 'fence',
                'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
            ],
            with_label=False),
        dict(
            type='Collect3D',
            keys=['img_inputs'],
            meta_keys=['pc_range', 'occ_size', 'sequence', 'frame_id'])
    ],
    classes=[
        'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
        'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
        'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
        'pole', 'traffic-sign'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    split='test-submit',
    camera_used=['left'],
    occ_size=[256, 256, 32],
    pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4])
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.01,
    eps=1e-08,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys=dict(
            query_embed=dict(lr_mult=1.0, decay_mult=0.0),
            query_feat=dict(lr_mult=1.0, decay_mult=0.0),
            level_embed=dict(lr_mult=1.0, decay_mult=0.0),
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0)),
        norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
lr_config = dict(policy='step', step=[20, 25])
runner = dict(type='EpochBasedRunner', max_epochs=30)
gpu_ids = range(0, 2)

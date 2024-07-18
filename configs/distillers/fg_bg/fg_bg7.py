_base_ = [
    '../../yolox/yolox_s_8x8_300e_coco.py'
]
# model settings
find_unused_parameters=True
temp=0.5
alpha_fgd=0.002
beta_fgd=0.001
gamma_fgd=0.001
lambda_fgd=0.00001
distiller = dict(
    type='DetectionDistiller',
    teacher_pretrained = 'work_dirs/best_bbox_mAP_epoch_120.pth',
    init_student = False,
    yolox = True,
    distill_cfg = [ dict(student_module = 'neck.out_convs.0',
                         teacher_module = 'neck.out_convs.0',
                         output_hook = True,
                         methods=[dict(type='fg_bgLoss',
                                       name='loss_fg_fpn_0',
                                       student_channels = 128,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       fg_bg_use=7
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.out_convs.1',
                         teacher_module = 'neck.out_convs.1',
                         output_hook = True,
                         methods=[dict(type='fg_bgLoss',
                                       name='loss_fg_fpn_1',
                                       student_channels = 128,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       fg_bg_use=7
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.out_convs.2',
                         teacher_module = 'neck.out_convs.2',
                         output_hook = True,
                         methods=[dict(type='fg_bgLoss',
                                       name='loss_fg_fpn_2',
                                       student_channels = 128,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       fg_bg_use=7
                                       )
                                ]
                        ),

                   ]
    )


student_cfg = 'configs/yolox/yolox_s_8x8_300e_coco.py'
teacher_cfg = 'configs/yolox/yolox_l_8x8_300e_coco.py'

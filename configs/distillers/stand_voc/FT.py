_base_ = [
    '../../yolox/yolox_s_voc.py'
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
    teacher_pretrained = 'work_dirs/best_mAP_epoch_75.pth',
    init_student = False,
    yolox = True,
    distill_cfg = [ dict(student_module = 'neck.out_convs.0',
                         teacher_module = 'neck.out_convs.0',
                         output_hook = True,
                         methods=[dict(type='FactorTransfer',
                                       name='loss_FT_fpn_0',
                             
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.out_convs.1',
                         teacher_module = 'neck.out_convs.1',
                         output_hook = True,
                         methods=[dict(type='FactorTransfer',
                                       name='loss_FT_fpn_1',
                                 
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.out_convs.2',
                         teacher_module = 'neck.out_convs.2',
                         output_hook = True,
                         methods=[dict(type='FactorTransfer',
                                       name='loss_FT_fpn_2',
                                    
                                       )
                                ]
                        ),

                   ]
    )


student_cfg = 'configs/yolox/yolox_s_voc.py'
teacher_cfg = 'configs/yolox/yolox_l_voc.py'
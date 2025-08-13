def add_common_arguments(parser):
    # ========== 原有参数 ==========
    parser.add_argument('--model_config', type=str, help='path to the model config file.')
    parser.add_argument('--dataset_config', type=str, help='path to the dataset config file.')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_variant', type=str, default='default')
    parser.add_argument('--output', type=str, required=True, help='out directory name.')
    parser.add_argument('--train_device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default='', help='path to the model to resume from.')
    parser.add_argument('--resume_epoch', type=int, default=-1,
        help='epoch to resume from when --resume flag provided.')
    parser.add_argument('-n', '--num_workers', type=int, default=0, help='Dataloader num_workers.')
    parser.add_argument('--log_interval',
        type=int, default=1, help='Log every this number of iterations.')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
    parser.add_argument('--dataset_type', type=str, default='costmap_4',
        help='Dataset type (mainly for visualization purposes). Could be "costmap_4" or "kitti_19"')
    parser.add_argument('--include_unknown', action='store_true', default=False,
                        help='Include the unknown class.')

    parser.add_argument('--buffer_scans', type=int, default=1,
                        help='How many scans to merge.')
    parser.add_argument('--buffer_scan_stride', type=int, default=1,
                        help='Stride between adjacent scans.')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--lr_decay_epoch', type=int, default=1,
                        help='Decay learning rate every this number of epochs.')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Learning rate decay.')
    
    # ========== 新增的渐进式训练参数 ==========
    
    # Deformable卷积渐进训练参数
    parser.add_argument('--progressive_deformable', action='store_true', default=False,
                        help='Enable progressive training for deformable convolutions. '
                             'Freezes offset predictors for initial epochs then unfreezes with lower lr.')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='Number of warmup epochs before unfreezing deformable layers.')
    parser.add_argument('--deform_lr_factor', type=float, default=0.1,
                        help='Learning rate multiplier for deformable offset/modulator parameters.')
    
    # 层级渐进训练参数
    parser.add_argument('--progressive_layers', action='store_true', default=False,
                        help='Enable layer-wise progressive training (encoder first, then decoder).')
    parser.add_argument('--encoder_only_epochs', type=int, default=2,
                        help='Train encoder only for these many initial epochs.')
    
    # 多尺度特征融合渐进训练（可选）
    parser.add_argument('--progressive_fusion', action='store_true', default=False,
                        help='Enable progressive multi-scale feature fusion.')
    parser.add_argument('--fusion_start_epoch', type=int, default=5,
                        help='Epoch to start using multi-scale features.')
    
    # 梯度裁剪参数
    parser.add_argument('--grad_clip', type=float, default=0.0,
                        help='Gradient clipping value. 0 means no clipping. '
                             'Recommended: 5.0 for deformable convolutions.')
    
    # 高级学习率调度策略
    parser.add_argument('--lr_schedule', type=str, default='step',
                        choices=['step', 'cosine', 'poly', 'warmup_cosine'],
                        help='Learning rate schedule type. '
                             '"step": StepLR (default), '
                             '"cosine": CosineAnnealingLR, '
                             '"poly": Polynomial decay, '
                             '"warmup_cosine": Linear warmup + Cosine annealing')
    parser.add_argument('--warmup_lr', type=float, default=1e-5,
                        help='Initial learning rate for warmup phase (only for warmup_cosine schedule).')
    
    # 调试和监控参数
    parser.add_argument('--verbose_training', action='store_true', default=False,
                        help='Enable verbose training output with detailed parameter statistics.')
    parser.add_argument('--save_checkpoint_every', type=int, default=0,
                        help='Save checkpoint every N epochs. 0 means only save best and final.')
    
    return parser
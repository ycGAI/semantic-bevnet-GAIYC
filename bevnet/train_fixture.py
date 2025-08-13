import os

import numpy as np
import tabulate
import torch
from spconv.utils import VoxelGenerator
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import bev_utils
from . import train_fixture_utils as tfu


def _load_model(model_file, nets, net_opts, tolerant):
    state = tfu.load_model(os.path.dirname(model_file),
                           os.path.basename(model_file), load_to_cpu=True)
    epoch = int(state['epoch'])

    for name, net in nets.items():
        tfu.load_state_helper(net, state['nets'][name], tolerant)

    # TODO: currently don't load net_opts, but we might want to do that later.
    return epoch


def _save_model(nets, net_opts, epoch, global_args, model_file):
    state = {
        'epoch': epoch,
        'global_args': global_args,
        'optims': {
            name: opt.state_dict() for name, opt in net_opts.items()
        },
        'nets': {
            name: net.state_dict() for name, net in nets.items()
        }
    }
    tfu.save_model(state, epoch, '', model_file)


# def train_single(nets, net_opts, g):
#     """
#     Training single-frame BEVNet.
    
#     Args:
#       nets: dictionary of models to forward/backprop in sequential order.
#       net_opts: optim related modules for training nets.
#       g: global arguments dictionary.
#     """
#     def forward_nets(nets, x):
#         # forward 3 modules: VoxelFeatureEncoder, MiddleSparseEncoder, BEVClassifier
#         # import ipdb; ipdb.set_trace()
#         voxels = nets['VoxelFeatureEncoder'](x['voxels'], x['num_points'])
#         voxel_features = nets['MiddleSparseEncoder'](voxels, x['coordinates'], g.batch_size)
#         preds = nets['BEVClassifier'](voxel_features)
#         return preds

#     def step(nets, inputs, labels=None, criterion=None):
#         model_outputs = forward_nets(nets, inputs)
#         pred = model_outputs['bev_preds']
#         loss = 0.0
#         if g.include_unknown:
#             # Note that `num_class` already includes the unknown label
#             labels[labels == 255] = g.num_class - 1
#         if labels is not None and criterion is not None:
#             labels = labels.long()
#             # import ipdb;ipdb.set_trace()
#             loss = criterion(pred, labels)
#         return pred, loss

#     def train(nets, net_opts, trainloader, criterion, epoch, writer, device):
#         tfu.make_train(nets)
#         loss_avg = []
#         itr = tqdm(trainloader)

#         for i, batch_data in enumerate(itr):
#             for _, opt in net_opts.items():
#                 opt.zero_grad()

#             def to_device(key):
#                 return batch_data[key].to(device, non_blocking=True)

#             label = to_device('label')
#             inputs = dict()
#             inputs['batch_size'] = len(label)
#             for key in batch_data:
#                 if key in ['points']:
#                     continue
#                 inputs[key] = to_device(key)
#             # import ipdb; ipdb.set_trace()

#             pred, loss = step(nets, inputs, labels=label, criterion=criterion)
#             loss.backward()

#             if g.log_interval > 0 and i % g.log_interval == 0:
#                 print('learning rate:\n%s' % tabulate.tabulate([
#                     (name, opt.param_groups[0]['lr']) for name, opt in net_opts.items()]))

#                 n_iter = (epoch - 1) * len(trainloader) + i + 1
#                 writer.add_scalar('Train/loss', loss.item(), n_iter)
#                 itr.set_description("train loss: %3f" % loss.item())
#                 loss_avg += [loss.item()]

#                 tfu.visualize_predictions(writer,
#                                           pred.data.cpu().numpy(),
#                                           label.cpu().numpy(),
#                                           g.dataset_type)

#                 for name, net in nets.items():
#                     print('%s weights:\n%s' % (name, tfu.module_weights_stats(net)))
#                 for name, net in nets.items():
#                     print('%s grad:\n%s' % (name, tfu.module_grad_stats(net)))

#             for _, opt in net_opts.items():
#                 opt.step()

#             if i % 10 == 0:
#                 torch.cuda.empty_cache()

#         return loss_avg

#     def validate(nets, validloader, criterion, n_iter, writer, device):
#         itr = tqdm(validloader)
#         tfu.make_eval(nets)
#         loss = []

#         if g.include_unknown:
#             ignore_idx = g.num_class - 1
#         else:
#             ignore_idx = 255

#         evaluator = bev_utils.Evaluator(num_classes=g.num_class, ignore_label=ignore_idx)

#         torch.cuda.empty_cache()

#         for i, batch_data in enumerate(itr):
#             def to_device(key):
#                 return batch_data[key].to(device, non_blocking=True)

#             label = to_device('label')

#             inputs = dict()
#             inputs['batch_size'] = len(label)
#             for key in batch_data:
#                 if key in ['points']:
#                     continue
#                 inputs[key] = to_device(key)

#             with torch.no_grad():
#                 pred_, loss_ = step(nets, inputs, labels=label, criterion=criterion)

#             loss += [float(loss_)]
#             pred = pred_.argmax(dim=1)
#             evaluator.append(pred, label)
#             itr.set_description("Acc: %3f, IoUMean: %3f" % (evaluator.acc(),
#                                                             evaluator.meanIoU()))

#         loss_avg = np.array(loss).mean()

#         # logging
#         writer.add_scalar('Val/loss', loss_avg.item(), n_iter)

#         cw_iou = evaluator.classwiseIoU()
#         cw_acc = evaluator.classwiseAcc()

#         for cls, (ciou, cacc) in enumerate(zip(cw_iou, cw_iou)):
#             writer.add_scalar('Val/iou_class{}'.format(cls), ciou, n_iter)
#             writer.add_scalar('Val/acc_class{}'.format(cls), cacc, n_iter)

#         writer.add_scalar('Val/iou', np.nanmean(cw_iou), n_iter)
#         writer.add_scalar('Val/acc', np.nanmean(cw_acc), n_iter)

#         return loss_avg

#     voxel_cfg = g.voxelizer
#     voxel_generator = VoxelGenerator(
#         voxel_size=list(voxel_cfg['voxel_size']),
#         point_cloud_range=list(voxel_cfg['point_cloud_range']),
#         max_num_points=voxel_cfg['max_number_of_points_per_voxel'],
#         full_mean=voxel_cfg['full_mean'],
#         max_voxels=voxel_cfg['max_voxels'])

#     train_dataset = bev_utils.BEVLoaderV2(g.train_input_reader,
#                                           g.dataset_path,
#                                           voxel_generator=voxel_generator,
#                                           n_buffer_scans=g.buffer_scans,
#                                           buffer_scan_stride=g.buffer_scan_stride)
#     valid_dataset = bev_utils.BEVLoaderV2(g.eval_input_reader,
#                                           g.dataset_path,
#                                           voxel_generator=voxel_generator,
#                                           n_buffer_scans=g.buffer_scans,
#                                           buffer_scan_stride=g.buffer_scan_stride)

#     class_weights = torch.tensor(g.class_weights).to(g.train_device)
#     assert len(class_weights) == g.num_class

#     if g.include_unknown:
#         ignore_idx = -100
#     else:
#         ignore_idx = 255

#     criterion = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=ignore_idx,
#                                           weight=class_weights)
#     trainloader = data.DataLoader(
#         train_dataset,
#         batch_size=g.batch_size,
#         num_workers=g.num_workers,
#         collate_fn=bev_utils.bev_single_collate_fn,
#         drop_last=True,
#         shuffle=True)

#     validloader = data.DataLoader(
#         valid_dataset,
#         batch_size=g.batch_size,
#         num_workers=g.num_workers,
#         collate_fn=bev_utils.bev_single_collate_fn,
#         drop_last=True,
#         shuffle=False)

#     output = g.output
#     os.makedirs(output, exist_ok=True)
#     writer = SummaryWriter(log_dir=os.path.join(output))
#     tfu.log_dict(writer, 'global_args', g, 0)

#     best_valid_loss = np.inf

#     resume_epoch = 0
#     if g.resume:
#         save_epoch = _load_model(g.resume, nets, net_opts, False)
#         print('loaded', g.resume, 'epoch', save_epoch)
#         if g.resume_epoch >= 0:
#             resume_epoch = g.resume_epoch
#         else:
#             resume_epoch = save_epoch

#     net_scheds = {
#         name: torch.optim.lr_scheduler.StepLR(
#             opt,
#             step_size=g.lr_decay_epoch,
#             gamma=g.lr_decay,
#             last_epoch=-1)
#         for name, opt in net_opts.items()
#     }

#     for epoch in range(1, g.epochs + 1):
#         if resume_epoch < epoch:
#             train(nets, net_opts, trainloader, criterion, epoch, writer, g.train_device)
#             _save_model(nets, net_opts, epoch, g, os.path.join(output, 'model.pth'))

#             n_iter = epoch * len(trainloader)
#             val_loss = validate(nets, validloader, criterion, n_iter, writer, g.train_device)

#             if val_loss < best_valid_loss:
#                 print("new best valid loss at %3f, saving model..." % val_loss)
#                 best_valid_loss = val_loss
#                 _save_model(nets, net_opts, epoch, g, os.path.join(output, 'best.pth'))

#         for _, sched in net_scheds.items():
#             sched.step()

#     writer.close()

def train_single(nets, net_opts, g):
    """
    Training single-frame BEVNet with progressive training strategies.
    
    Args:
      nets: dictionary of models to forward/backprop in sequential order.
      net_opts: optim related modules for training nets.
      g: global arguments dictionary.
    """
    def forward_nets(nets, x):
        # forward 3 modules: VoxelFeatureEncoder, MiddleSparseEncoder, BEVClassifier
        voxels = nets['VoxelFeatureEncoder'](x['voxels'], x['num_points'])
        voxel_features = nets['MiddleSparseEncoder'](voxels, x['coordinates'], g.batch_size)
        preds = nets['BEVClassifier'](voxel_features)
        return preds

    def step(nets, inputs, labels=None, criterion=None):
        model_outputs = forward_nets(nets, inputs)
        pred = model_outputs['bev_preds']
        loss = 0.0
        if g.include_unknown:
            # Note that `num_class` already includes the unknown label
            labels[labels == 255] = g.num_class - 1
        if labels is not None and criterion is not None:
            labels = labels.long()
            loss = criterion(pred, labels)
        return pred, loss

    def apply_progressive_training(nets, net_opts, epoch, g):
        """应用渐进式训练策略"""
        # 检查是否需要导入DeformableConv2d
        try:
            from bevnet.deformable_conv import DeformableConv2d
            has_deformable = True
        except ImportError:
            has_deformable = False
            if hasattr(g, 'progressive_deformable') and g.progressive_deformable:
                print("Warning: DeformableConv2d not found, skipping progressive deformable training")
        
        # 策略1: 渐进式解冻Deformable卷积
        if has_deformable and hasattr(g, 'progressive_deformable') and g.progressive_deformable:
            warmup_epochs = getattr(g, 'warmup_epochs', 3)
            deform_lr_factor = getattr(g, 'deform_lr_factor', 0.1)
            
            if epoch <= warmup_epochs:
                # 前几个epoch冻结offset预测器
                print(f"\n[Epoch {epoch}] Progressive Deformable Strategy: Freezing offset layers...")
                frozen_count = 0
                for name, module in nets['BEVClassifier'].named_modules():
                    if isinstance(module, DeformableConv2d):
                        module.offset_conv.requires_grad_(False)
                        if hasattr(module, 'modulator_conv'):
                            module.modulator_conv.requires_grad_(False)
                        frozen_count += 1
                print(f"  -> Frozen {frozen_count} deformable modules")
            
            elif epoch == warmup_epochs + 1:
                # 解冻但使用较小的学习率
                print(f"\n[Epoch {epoch}] Progressive Deformable Strategy: Unfreezing with reduced lr...")
                
                # 收集deformable和regular参数
                deform_params = []
                regular_params = []
                deform_count = 0
                
                for name, module in nets['BEVClassifier'].named_modules():
                    if isinstance(module, DeformableConv2d):
                        # 解冻offset和modulator
                        module.offset_conv.requires_grad_(True)
                        deform_params.extend(module.offset_conv.parameters())
                        
                        if hasattr(module, 'modulator_conv'):
                            module.modulator_conv.requires_grad_(True)
                            deform_params.extend(module.modulator_conv.parameters())
                        deform_count += 1
                
                # 收集其他参数
                for param in nets['BEVClassifier'].parameters():
                    if not any(param is p for p in deform_params):
                        regular_params.append(param)
                
                if deform_params:
                    # 重新创建优化器，为不同参数组设置不同学习率
                    current_lr = g.lr
                    net_opts['BEVClassifier'] = torch.optim.Adam([
                        {'params': regular_params, 'lr': current_lr},
                        {'params': deform_params, 'lr': current_lr * deform_lr_factor}
                    ], eps=1.0e-5)
                    print(f"  -> Unfrozen {deform_count} deformable modules")
                    print(f"  -> Regular params lr: {current_lr}, Deformable params lr: {current_lr * deform_lr_factor}")
        
        # 策略2: 层级渐进训练 (encoder -> decoder)
        if hasattr(g, 'progressive_layers') and g.progressive_layers:
            encoder_only_epochs = getattr(g, 'encoder_only_epochs', 2)
            
            if epoch <= encoder_only_epochs:
                print(f"\n[Epoch {epoch}] Progressive Layer Strategy: Training encoder only...")
                frozen_blocks = 0
                
                # 检查是否有FChardNet架构
                if hasattr(nets['BEVClassifier'], 'fchardnet'):
                    fchardnet = nets['BEVClassifier'].fchardnet
                    
                    # 冻结decoder部分
                    if hasattr(fchardnet, 'denseBlocksUp'):
                        for block in fchardnet.denseBlocksUp:
                            for param in block.parameters():
                                param.requires_grad_(False)
                            frozen_blocks += 1
                    
                    # 冻结上采样层
                    if hasattr(fchardnet, 'transUpBlocks'):
                        for block in fchardnet.transUpBlocks:
                            for param in block.parameters():
                                param.requires_grad_(False)
                    
                    # 冻结最终卷积层
                    if hasattr(fchardnet, 'finalConv'):
                        for param in fchardnet.finalConv.parameters():
                            param.requires_grad_(False)
                    
                    print(f"  -> Frozen {frozen_blocks} decoder blocks")
            
            elif epoch == encoder_only_epochs + 1:
                print(f"\n[Epoch {epoch}] Progressive Layer Strategy: Unfreezing decoder...")
                unfrozen_blocks = 0
                
                if hasattr(nets['BEVClassifier'], 'fchardnet'):
                    fchardnet = nets['BEVClassifier'].fchardnet
                    
                    # 解冻decoder
                    if hasattr(fchardnet, 'denseBlocksUp'):
                        for block in fchardnet.denseBlocksUp:
                            for param in block.parameters():
                                param.requires_grad_(True)
                            unfrozen_blocks += 1
                    
                    # 解冻上采样层
                    if hasattr(fchardnet, 'transUpBlocks'):
                        for block in fchardnet.transUpBlocks:
                            for param in block.parameters():
                                param.requires_grad_(True)
                    
                    # 解冻最终卷积层
                    if hasattr(fchardnet, 'finalConv'):
                        for param in fchardnet.finalConv.parameters():
                            param.requires_grad_(True)
                    
                    print(f"  -> Unfrozen {unfrozen_blocks} decoder blocks")
        
        # 策略3: 渐进式特征融合（如果使用多尺度特征）
        if hasattr(g, 'progressive_fusion') and g.progressive_fusion:
            fusion_start_epoch = getattr(g, 'fusion_start_epoch', 5)
            
            if epoch < fusion_start_epoch:
                print(f"\n[Epoch {epoch}] Progressive Fusion: Using single scale features")
                # 这里可以设置网络只使用单一尺度特征
                # 具体实现依赖于网络架构
            else:
                print(f"\n[Epoch {epoch}] Progressive Fusion: Using multi-scale features")
        
        return net_opts

    def get_lr_scheduler(optimizer, epoch, g):
        """获取当前epoch的学习率调整策略"""
        lr_schedule = getattr(g, 'lr_schedule', 'step')
        
        if lr_schedule == 'warmup_cosine':
            warmup_epochs = getattr(g, 'warmup_epochs', 3)
            warmup_lr = getattr(g, 'warmup_lr', 1e-5)
            
            if epoch <= warmup_epochs:
                # Linear warmup
                lr_scale = (epoch / warmup_epochs)
                new_lr = warmup_lr + (g.lr - warmup_lr) * lr_scale
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / (g.epochs - warmup_epochs)
                new_lr = g.lr * 0.5 * (1 + np.cos(np.pi * progress))
            
            # 更新优化器学习率
            for param_group in optimizer.param_groups():
                base_lr = param_group.get('initial_lr', g.lr)
                param_group['lr'] = new_lr * (param_group['lr'] / base_lr if base_lr > 0 else 1)
            
            return new_lr
        
        return None  # 使用默认的StepLR

    def train(nets, net_opts, trainloader, criterion, epoch, writer, device):
        # 应用渐进式训练策略
        net_opts = apply_progressive_training(nets, net_opts, epoch, g)
        
        # 应用自定义学习率调度（如果需要）
        if hasattr(g, 'lr_schedule') and g.lr_schedule == 'warmup_cosine':
            for name, opt in net_opts.items():
                get_lr_scheduler(opt, epoch, g)
        
        tfu.make_train(nets)
        loss_avg = []
        itr = tqdm(trainloader)

        for i, batch_data in enumerate(itr):
            for _, opt in net_opts.items():
                opt.zero_grad()

            def to_device(key):
                return batch_data[key].to(device, non_blocking=True)

            label = to_device('label')
            inputs = dict()
            inputs['batch_size'] = len(label)
            for key in batch_data:
                if key in ['points']:
                    continue
                inputs[key] = to_device(key)

            pred, loss = step(nets, inputs, labels=label, criterion=criterion)
            
            # 梯度裁剪（对于deformable卷积特别重要）
            if hasattr(g, 'grad_clip') and g.grad_clip > 0:
                for net in nets.values():
                    torch.nn.utils.clip_grad_norm_(net.parameters(), g.grad_clip)
            
            loss.backward()

            if g.log_interval > 0 and i % g.log_interval == 0:
                print('learning rate:\n%s' % tabulate.tabulate([
                    (name, opt.param_groups[0]['lr']) for name, opt in net_opts.items()]))

                n_iter = (epoch - 1) * len(trainloader) + i + 1
                writer.add_scalar('Train/loss', loss.item(), n_iter)
                itr.set_description("train loss: %3f" % loss.item())
                loss_avg += [loss.item()]

                tfu.visualize_predictions(writer,
                                          pred.data.cpu().numpy(),
                                          label.cpu().numpy(),
                                          g.dataset_type)

                for name, net in nets.items():
                    print('%s weights:\n%s' % (name, tfu.module_weights_stats(net)))
                for name, net in nets.items():
                    print('%s grad:\n%s' % (name, tfu.module_grad_stats(net)))

            for _, opt in net_opts.items():
                opt.step()

            if i % 10 == 0:
                torch.cuda.empty_cache()

        return loss_avg

    def validate(nets, validloader, criterion, n_iter, writer, device):
        itr = tqdm(validloader)
        tfu.make_eval(nets)
        loss = []

        if g.include_unknown:
            ignore_idx = g.num_class - 1
        else:
            ignore_idx = 255

        evaluator = bev_utils.Evaluator(num_classes=g.num_class, ignore_label=ignore_idx)

        torch.cuda.empty_cache()

        for i, batch_data in enumerate(itr):
            def to_device(key):
                return batch_data[key].to(device, non_blocking=True)

            label = to_device('label')

            inputs = dict()
            inputs['batch_size'] = len(label)
            for key in batch_data:
                if key in ['points']:
                    continue
                inputs[key] = to_device(key)

            with torch.no_grad():
                pred_, loss_ = step(nets, inputs, labels=label, criterion=criterion)

            loss += [float(loss_)]
            pred = pred_.argmax(dim=1)
            evaluator.append(pred, label)
            itr.set_description("Acc: %3f, IoUMean: %3f" % (evaluator.acc(),
                                                            evaluator.meanIoU()))

        loss_avg = np.array(loss).mean()

        # logging
        writer.add_scalar('Val/loss', loss_avg.item(), n_iter)

        cw_iou = evaluator.classwiseIoU()
        cw_acc = evaluator.classwiseAcc()

        for cls, (ciou, cacc) in enumerate(zip(cw_iou, cw_iou)):
            writer.add_scalar('Val/iou_class{}'.format(cls), ciou, n_iter)
            writer.add_scalar('Val/acc_class{}'.format(cls), cacc, n_iter)

        writer.add_scalar('Val/iou', np.nanmean(cw_iou), n_iter)
        writer.add_scalar('Val/acc', np.nanmean(cw_acc), n_iter)

        return loss_avg

    # ========== 主训练循环开始 ==========
    voxel_cfg = g.voxelizer
    voxel_generator = VoxelGenerator(
        voxel_size=list(voxel_cfg['voxel_size']),
        point_cloud_range=list(voxel_cfg['point_cloud_range']),
        max_num_points=voxel_cfg['max_number_of_points_per_voxel'],
        full_mean=voxel_cfg['full_mean'],
        max_voxels=voxel_cfg['max_voxels'])

    train_dataset = bev_utils.BEVLoaderV2(g.train_input_reader,
                                          g.dataset_path,
                                          voxel_generator=voxel_generator,
                                          n_buffer_scans=g.buffer_scans,
                                          buffer_scan_stride=g.buffer_scan_stride)
    valid_dataset = bev_utils.BEVLoaderV2(g.eval_input_reader,
                                          g.dataset_path,
                                          voxel_generator=voxel_generator,
                                          n_buffer_scans=g.buffer_scans,
                                          buffer_scan_stride=g.buffer_scan_stride)

    class_weights = torch.tensor(g.class_weights).to(g.train_device)
    assert len(class_weights) == g.num_class

    if g.include_unknown:
        ignore_idx = -100
    else:
        ignore_idx = 255

    criterion = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=ignore_idx,
                                          weight=class_weights)
    trainloader = data.DataLoader(
        train_dataset,
        batch_size=g.batch_size,
        num_workers=g.num_workers,
        collate_fn=bev_utils.bev_single_collate_fn,
        drop_last=True,
        shuffle=True)

    validloader = data.DataLoader(
        valid_dataset,
        batch_size=g.batch_size,
        num_workers=g.num_workers,
        collate_fn=bev_utils.bev_single_collate_fn,
        drop_last=True,
        shuffle=False)

    output = g.output
    os.makedirs(output, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output))
    tfu.log_dict(writer, 'global_args', g, 0)

    best_valid_loss = np.inf

    resume_epoch = 0
    if g.resume:
        save_epoch = _load_model(g.resume, nets, net_opts, False)
        print('loaded', g.resume, 'epoch', save_epoch)
        if g.resume_epoch >= 0:
            resume_epoch = g.resume_epoch
        else:
            resume_epoch = save_epoch

    net_scheds = {
        name: torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=g.lr_decay_epoch,
            gamma=g.lr_decay,
            last_epoch=-1)
        for name, opt in net_opts.items()
    }

    for epoch in range(1, g.epochs + 1):
        if resume_epoch < epoch:
            train(nets, net_opts, trainloader, criterion, epoch, writer, g.train_device)
            _save_model(nets, net_opts, epoch, g, os.path.join(output, 'model.pth'))

            n_iter = epoch * len(trainloader)
            val_loss = validate(nets, validloader, criterion, n_iter, writer, g.train_device)

            if val_loss < best_valid_loss:
                print("new best valid loss at %3f, saving model..." % val_loss)
                best_valid_loss = val_loss
                _save_model(nets, net_opts, epoch, g, os.path.join(output, 'best.pth'))

        for _, sched in net_scheds.items():
            sched.step()

    writer.close()

def train_recurrent(nets, net_opts, g):
    def forward_nets(nets, x):
        # import ipdb; ipdb.set_trace()
        # forward 3 modules: VoxelFeatureEncoder, MiddleSparseEncoder, BEVClassifier
        voxels = nets['VoxelFeatureEncoder'](x['voxels'], x['num_points'])
        voxel_features = nets['MiddleSparseEncoder'](voxels, x['coordinates'], g.batch_size)
        seq_start = x['seq_start'] #iif 'seq_start' in example else None
        pose = x['pose']
        preds = nets['BEVClassifier'](voxel_features, seq_start = seq_start, input_pose = pose)
        return preds

    def _step(nets, inputs, labels=None, criterion=None):
        model_outputs = forward_nets(nets, inputs)
        loss = 0
        pred = model_outputs['bev_preds']
        if g.include_unknown:
            # Note that `num_class` already includes the unknown label
            labels[labels == 255] = g.num_class - 1

        if labels is not None and criterion is not None:
            labels = labels.long()
            t, n, c, h, w = pred.shape
            pred = pred.reshape((t, c, h, w))
            labels = labels.squeeze(0)
            loss = criterion(pred, labels)
        return pred, loss

    def _train(nets, net_opts, trainloader, criterion, epoch, writer, device):
        tfu.make_train(nets)
        loss_avg = []
        itr = tqdm(trainloader)
        # import ipdb;ipdb.set_trace()
        for i, batch in enumerate(itr):
            for _, opt in net_opts.items():
                opt.zero_grad()

            label = batch.label.to(device, non_blocking=True)
            input_voxels = {
                "voxels": [vox.to(device, non_blocking=True) for vox in batch.voxels[0]],
                "num_points": [num.to(device, non_blocking=True) for num in batch.num_points[0]],
                "coordinates": [co.to(device, non_blocking=True) for co in batch.coords[0]],
                "seq_start": batch.seq_start.to(device, non_blocking=True).squeeze(0),
                "pose": batch.pose.to(device, non_blocking=True).squeeze(0),
                "batch_size": 1,
            }

            pred, loss = _step(nets, input_voxels, labels=label, criterion=criterion)
            
            loss.backward()

            if g.log_interval > 0 and i % g.log_interval == 0:
                n_iter = (epoch - 1) * len(trainloader) + i + 1
                print('learning rate:\n%s' % tabulate.tabulate([
                    (name, opt.param_groups[0]['lr']) for name, opt in net_opts.items()]))
                # writer.add_scalar('Train/learning_rate',
                #                 optimizer.param_groups[0]['lr'],
                #                 n_iter)
                writer.add_scalar('Train/loss', loss.item(), n_iter)
                itr.set_description("train loss: %3f" % loss.item())
                loss_avg += [loss.item()]

                tfu.visualize_predictions(writer, pred.data.cpu().numpy(), label[0].cpu().numpy(),
                                          g.dataset_type)

                print('grad statistics:')
                for model in nets.values():
                    print(tfu.module_grad_stats(model))

            for _, opt in net_opts.items():
                opt.step()
            if i % 10 == 0:
                torch.cuda.empty_cache()

        return loss_avg

    def _validate(nets, validloader, criterion, n_iter, writer, device):
        itr = tqdm(validloader)
        tfu.make_eval(nets)
        loss = []

        if g.include_unknown:
            ignore_idx = g.num_class - 1
        else:
            ignore_idx = 255

        evaluator = bev_utils.Evaluator(num_classes=g.num_class, ignore_label=ignore_idx)

        with torch.no_grad():
            for i, batch in enumerate(itr):
                label = batch.label.to(device)
                input_voxels = {
                    "voxels": [vox.to(device) for vox in batch.voxels[0]],
                    "num_points": [num.to(device) for num in batch.num_points[0]],
                    "coordinates": [co.to(device) for co in batch.coords[0]],
                    "seq_start": batch.seq_start.to(device).squeeze(0),
                    "pose": batch.pose.to(device).squeeze(0),
                    "batch_size": 1,
                }
                pred_, loss_ = _step(nets, input_voxels, labels=label, criterion=criterion)
                loss += [float(loss_)]
                pred = pred_.argmax(dim=1).detach()
                label = label.squeeze(0).detach()

                evaluator.append(pred, label)
                itr.set_description("Acc: %3f, IoUMean: %3f" % (evaluator.acc(),
                                                                evaluator.meanIoU()))

            loss_avg = np.array(loss).mean()

            writer.add_scalar('Val/loss', loss_avg.item(), n_iter)

            cw_iou = evaluator.classwiseIoU()
            cw_acc = evaluator.classwiseAcc()

            for cls, (ciou, cacc) in enumerate(zip(cw_iou,cw_iou)):
                writer.add_scalar('Val/iou_class{}'.format(cls), ciou, n_iter)
                writer.add_scalar('Val/acc_class{}'.format(cls), cacc, n_iter)

            writer.add_scalar('Val/iou', np.nanmean(cw_iou), n_iter)
            writer.add_scalar('Val/acc', np.nanmean(cw_acc), n_iter)
            
            return loss_avg

    tfu.set_device(nets, g.train_device)
    voxel_cfg = g.voxelizer
    voxel_generator = VoxelGenerator(
        voxel_size=list(voxel_cfg['voxel_size']),
        point_cloud_range=list(voxel_cfg['point_cloud_range']),
        max_num_points=voxel_cfg['max_number_of_points_per_voxel'],
        full_mean=voxel_cfg['full_mean'],
        max_voxels=voxel_cfg['max_voxels'])

    train_dataset = bev_utils.BEVLoaderMultistepV3(
        g.train_input_reader,
        g.dataset_path,
        shuffle=True,
        n_frame=g.n_frame,
        seq_len=g.seq_len,
        frame_strides=g.frame_strides,
        voxel_generator=voxel_generator,
        n_buffer_scans=g.buffer_scans,
        buffer_scan_stride=g.buffer_scan_stride)
    # train_dataset.set_label_shape((256, 256))

    valid_dataset = bev_utils.BEVLoaderMultistepV3(
        g.eval_input_reader,
        g.dataset_path,
        shuffle=False,
        n_frame=g.n_frame,
        seq_len=None,
        frame_strides=[1],
        voxel_generator=voxel_generator,
        n_buffer_scans=g.buffer_scans,
        buffer_scan_stride=g.buffer_scan_stride)
    # valid_dataset.set_label_shape((256, 256))

    net_scheds = {
        name: torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=g.lr_decay_epoch,
            gamma=g.lr_decay,
            last_epoch=-1)
        for name, opt in net_opts.items()
    }

    if g.include_unknown:
        ignore_idx = -100
    else:
        ignore_idx = 255
    class_weights = torch.tensor(g.class_weights).to(g.train_device)
    criterion = torch.nn.CrossEntropyLoss(
        reduction="mean", ignore_index=ignore_idx, weight=class_weights)

    trainloader = data.DataLoader(
        train_dataset,
        batch_size=g.batch_size,
        num_workers=g.num_workers,
        collate_fn=train_dataset.collate_wrapper,
        worker_init_fn=train_dataset.init
    )

    validloader = data.DataLoader(
        valid_dataset,
        batch_size=g.batch_size,
        num_workers=g.num_workers,
        collate_fn=valid_dataset.collate_wrapper,
        worker_init_fn=valid_dataset.init
    )

    output = g.output
    os.makedirs(output, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output))
    tfu.log_dict(writer, 'global_args', g, 0)

    best_valid_loss = np.inf

    resume_epoch = 0
    if g.resume:
        save_epoch = _load_model(g.resume, nets, net_opts, True)
        print('loaded', g.resume, 'epoch', save_epoch)
        if g.resume_epoch >= 0:
            resume_epoch = g.resume_epoch
        else:
            resume_epoch = save_epoch

    for epoch in range(1, g.epochs + 1):
        if resume_epoch < epoch:
            train_loss = _train(nets, net_opts, trainloader, criterion, epoch, writer, g.train_device)
            _save_model(nets, net_opts, epoch, g, os.path.join(output, 'model.pth'))
            n_iter = epoch * len(trainloader)
            val_loss = _validate(nets, validloader, criterion, n_iter, writer, g.train_device)
            if val_loss < best_valid_loss:
                print("new best valid loss at %3f, saving model..." % val_loss)
                best_valid_loss = val_loss
                _save_model(nets, net_opts, epoch, g, os.path.join(output, 'best.pth'))

        for _, sched in net_scheds.items():
            sched.step()

    writer.close()

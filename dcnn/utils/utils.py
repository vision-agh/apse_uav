import os
import sys
import torch
import copy
import matplotlib.pyplot as plt


def build_finetune_optimizer(cfg, model, to_train=['backbone', 'proposal_generator', 'roi_heads']):

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    defaults = {}
    defaults["lr"] = cfg.SOLVER.BASE_LR

    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()

    if 'backbone' in to_train:
        for module in model.backbone.modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                # print('\t', 'module param name:', module_param_name)
                
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if isinstance(module, norm_module_types) and cfg.SOLVER.WEIGHT_DECAY_NORM is not None:
                    hyperparams["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY_NORM
                params.append({"params": [value], **hyperparams})

    if 'proposal_generator' in to_train:
        for module in model.proposal_generator.modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                # print('\t', 'module param name:', module_param_name)
                
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if isinstance(module, norm_module_types) and cfg.SOLVER.WEIGHT_DECAY_NORM is not None:
                    hyperparams["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY_NORM
                params.append({"params": [value], **hyperparams})

    if 'roi_heads' in to_train:
        for module in model.roi_heads.modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                # print('\t', 'module param name:', module_param_name)
                
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if isinstance(module, norm_module_types) and cfg.SOLVER.WEIGHT_DECAY_NORM is not None:
                    hyperparams["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY_NORM
                params.append({"params": [value], **hyperparams})


    optimizer = torch.optim.SGD(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        nesterov=cfg.SOLVER.NESTEROV,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    return optimizer

def plot_training_results(results_filepath, output_path, extra_plot=False):

    epochs = []
    precisions = []
    recalls = []
    train_losses = []
    extra_precisions = []
    extra_recalls = []

    with open(results_filepath, 'r') as results_file:
        file_lines = results_file.readlines()
    
    for linedata in file_lines[1:]:
        linedata = linedata.split()
        epochs.append(int(linedata[0].split('/')[0]))
        precisions.append(float(linedata[1]))
        recalls.append(float(linedata[9]))
        train_losses.append(float(linedata[13]))
        if extra_plot:
            extra_precisions.append(float(linedata[14]))
            extra_recalls.append(float(linedata[15]))

    plt.figure(figsize=(10, 10))

    plt.subplot(211)
    plt.plot(epochs, precisions, 'green')
    plt.plot(epochs, recalls, 'red')
    if extra_plot:
        plt.plot(epochs, extra_precisions, 'g--')
        plt.plot(epochs, extra_recalls, 'r--')
    plt.xlabel('iteration')
    plt.ylabel('fitness')
    plt.grid(True)
    plt.legend(['Precision', 'Recall'])

    plt.subplot(212)
    plt.plot(epochs, train_losses)
    plt.ylabel('Total train loss')
    plt.xlabel('iteration')
    plt.grid(True)

    # # plt.show()
    plt.savefig(os.path.join(output_path, 'test_plot.png'))
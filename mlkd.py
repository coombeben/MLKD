import json
import warnings
import re
from collections import OrderedDict
from argparse import ArgumentParser
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel as parallel
import torch.optim as optim
import torch.utils.data as data
import timm

from tqdm import tqdm
from timm.data.mixup import Mixup

from utils.boilerplate import init_dist, gather_object
from utils.checkpoint import History
from utils.datasets import DataContainer, get_dataset, get_transforms_test, get_transforms_train, get_num_classes
from utils.ensemble import Ensemble
from utils.resnet import MultiResNetV2

warnings.simplefilter('ignore', UserWarning)

parser = ArgumentParser()
parser.add_argument('--world-size', default=1, type=int, help='number of gpus')
parser.add_argument('-d', '--dataset', default='oxford-iiit', type=str,
                    choices=['cifar-10', 'cifar-100', 'oxfird-iiit', 'dogs'], help='dataset choice')
parser.add_argument('-b', '--batch-size', default=64, type=int, help='batch size')
parser.add_argument('-e', '--epochs', default=1000, type=int, help='epochs to train for')
parser.add_argument('-n', '--name', default='mlkd_student', type=str, help='name to give saved model')
parser.add_argument('-c', '--checkpoint', help='path to teacher checkpoint')
parser.add_argument('-m', '--model', help='architecture of teacher model')
parser.add_argument('--mixup-alpha', default=1., type=float)
parser.add_argument('--cutmix-alpha', default=1., type=float)
parser.add_argument('--alpha', default=1., type=float, help='alpha of loss function')
parser.add_argument('--temperature', default=10., type=float, help='temperature of loss function')
parser.add_argument('--ensemble', default='', type=str, help='path to json config for ensemble')
parser.add_argument('-s', '--save-frequency', default=50, type=int, help='save every n epochs')

torch.manual_seed(53303)
torch.use_deterministic_algorithms(True, warn_only=True)


def save_model(model: nn.Module, path: str, distributed: bool = False):
    """Saves model, allowing for the fact the DDP doesn't allow custom methods"""
    if not distributed:
        model.save(path)
        return

    state_dict = OrderedDict()

    for key, value in model.module.state_dict().items():
        if not key.startswith(('bottleneck', 'avgpool', 'middle')):
            new_key = re.sub(r'stages(\d)', r'stages.\1', key)
            state_dict[new_key] = value

    torch.save(state_dict, path)


class TeachLoss(nn.Module):
    def __init__(self, temperature: float, alpha: float):
        """Loss function from Hinton et al.
        Uses a weighted mean of the distillation and student losses
        loss = alpha * distillation_loss + (1 - alpha) * student loss
        """
        assert 0 <= alpha <= 1
        super(TeachLoss, self).__init__()

        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits: torch.Tensor, teacher_softmax: torch.Tensor, target: torch.Tensor):
        student_log_softmax = F.log_softmax(student_logits / self.temperature, dim=1)
        distillation_loss = self.temperature * self.temperature * nn.KLDivLoss()(student_log_softmax, teacher_softmax)
        student_loss = nn.CrossEntropyLoss()(student_logits, target)

        return self.alpha * distillation_loss + (1 - self.alpha) * student_loss


def distil(train_loader: data.DataLoader, t_model: nn.Module, s_model: nn.Module, loss_fn: nn.Module,
                 optimiser: optim.Optimizer, rank: int, mixup: Callable, temperature: float) -> float:
    """Applies MLKD"""
    master_process = rank == 0
    accumulation_steps = max(1, 128 // train_loader.batch_size)

    t_model.eval()
    s_model.train()

    running_loss = 0

    for batch, (X, y) in enumerate(tqdm(train_loader, disable=not master_process)):
        X, y = X.to(rank), y.to(rank)
        X, y = mixup(X, y)

        with torch.no_grad():
            teacher_logits = t_model(X)
        student_logits, middle_output1, middle_output2, middle_output3 = s_model(X)

        teacher_softmax = F.softmax(teacher_logits / temperature, dim=1)

        loss = loss_fn(student_logits, teacher_softmax, y)

        middle1_loss = loss_fn(middle_output1, teacher_softmax, y)
        middle2_loss = loss_fn(middle_output2, teacher_softmax, y)
        middle3_loss = loss_fn(middle_output3, teacher_softmax, y)

        total_loss = loss + middle1_loss + middle2_loss + middle3_loss
        total_loss.backward()

        if (batch + 1) % accumulation_steps == 0 or batch + 1 == len(train_loader):
            optimiser.step()
            optimiser.zero_grad()

        running_loss += total_loss.item()

    running_loss /= len(train_loader)
    if master_process:
        print(f'Training Loss: {running_loss:>5f}')

    return running_loss


def validate(valid_loader: data.DataLoader, t_model: nn.Module, s_model: nn.Module, loss_fn,
             rank: int, temperature: float) -> (float, float):
    """Validation loop. Modified to allow model(X) to produce multiple outputs"""
    master_process = rank == 0
    running_loss, correct = 0, 0

    t_model.eval()
    s_model.eval()

    with torch.no_grad():
        for batch, (X, y) in enumerate(valid_loader):
            X, y = X.to(rank), y.to(rank)

            teacher_logits = t_model(X)
            student_logits, *_ = s_model(X)
            teacher_softmax = F.softmax(teacher_logits / temperature, dim=1)
            loss = loss_fn(student_logits, teacher_softmax, y)

            y_pred, y_true = torch.argmax(student_logits, dim=1), y.long().squeeze()
            correct += (y_pred == y_true).type(torch.float).sum().item()

            running_loss += loss.item()

    # Calculate accuracy and loss
    gather_correct = gather_object(correct)
    total_correct = sum(gather_correct) / len(valid_loader.dataset)
    running_loss /= len(valid_loader)

    if master_process:
        print(f"Validation Accuracy: {(100 * total_correct):>0.2f}%, Validation Loss: {running_loss:>5f}")

    return running_loss, total_correct


def main(rank: int, args):
    master_process = rank == 0
    if args.distributed:
        init_dist(rank, args.world_size)

    num_classes = get_num_classes(args.dataset)

    # Init model
    if args.ensemble:
        with open(args.ensemble, 'r') as json_file:
            ensemble_config_dict = json.load(json_file)
        teacher_model = Ensemble(ensemble_config_dict, num_classes=num_classes)
    else:
        teacher_model = timm.create_model(
            args.model, checkpoint_path=args.checkpoint, num_classes=num_classes)

    model = MultiResNetV2(num_classes=num_classes)

    # Init data transforms
    transforms_train = get_transforms_train()
    transforms_test = get_transforms_test()
    if not (args.mixup_alpha == 0 and args.cutmix_alpha == 0):
        mixup = Mixup(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            num_classes=num_classes
        )
    else:
        mixup = lambda *args: args

    # Init dataset
    train_dataset, valid_dataset, _ = get_dataset(
        args.dataset, transforms_train=transforms_train, transforms_test=transforms_test)
    dataset = DataContainer(rank, args.world_size, args.batch_size, train_dataset, valid_dataset)

    # Set models to GPU
    teacher_model.to(rank)
    model.to(rank)
    if args.distributed:
        model = parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    # Init training modules
    loss_fn = TeachLoss(args.temperature, args.alpha)
    optimiser = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_loss = torch.inf
    history = History()
    history.set_stage('distil', ['sd_loss', 'valid_loss', 'valid_acc'])

    # Distillation loop
    for epoch in range(args.epochs):
        if args.distributed:
            dataset.train_sampler.set_epoch(epoch)
            dataset.valid_sampler.set_epoch(epoch)

        # Do one pass of distillation and validation
        sd_loss = distil(
            dataset.train_dataloader, teacher_model, model, loss_fn, optimiser, rank, mixup, args.temperature)
        valid_loss, valid_acc = validate(
            dataset.valid_dataloader, teacher_model, model, loss_fn, rank, args.temperature)
        history.save(sd_loss, valid_loss, valid_acc)

        if master_process:
            history.export(args.name)

        # If model has improved, save checkpoint
        if args.distributed:
            dist.barrier()
        if valid_loss < best_loss:
            best_loss = valid_loss
            if master_process:
                save_model(model, f'models/{args.name}.pth', args.distributed)

        # Intermittently save checkpoints
        if (epoch + 1) % args.save_frequency == 0 and master_process:
            save_model(model, f'models/{args.name}_e_{epoch + 1}.pth', args.distributed)

    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    if args.world_size > 1:
        mp.spawn(main, args=(args,), nprocs=args.world_size, join=True)
    else:
        main(0, args)

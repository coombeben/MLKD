"""
Code for building stacked ensemble meta-learner
"""
import json
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.datasets import get_loader
from utils.ensemble import Ensemble
from utils.datasets import get_dataset, get_num_classes, get_transforms_train

torch.manual_seed(53303)
torch.use_deterministic_algorithms(True, warn_only=True)

parser = ArgumentParser()
parser.add_argument('-b', '--batch-size', default=64, type=int, help='batch size')
parser.add_argument('-d', '--dataset', default='oxford-iiit', type=str,
                    choices=['cifar-10', 'cifar-100', 'oxfird-iiit', 'dogs'], help='dataset choice')
parser.add_argument('-e', '--epochs', default=15, type=int, help='epochs')
parser.add_argument('--ensemble', type=str, required=True, help='path to json config for ensemble')


class StackModel(nn.Module):
    def __init__(self, n_models: int):
        super().__init__()
        self.fc = nn.Linear(n_models, 1, bias=False)

    def forward(self, inputs: torch.Tensor):
        return self.fc(inputs).squeeze()


def main(rank: int, args):
    with open(args.ensemble, 'r') as json_file:
        config = json.load(json_file)

    num_classes = get_num_classes(args.dataset)
    num_models = len(config['models'])

    model = Ensemble(config, num_classes)
    stack_model = StackModel(num_models)
    transforms_train = get_transforms_train()

    model.to(rank)
    stack_model.to(rank)

    dataset, _, _ = get_dataset(
        args.dataset, train=True, valid=False, test=False, transforms_train=transforms_train)
    data_loader, _ = get_loader(dataset, 1, rank, args.batch_size)

    optimiser = optim.AdamW(stack_model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    model.eval()
    stack_model.train()

    accumulation_steps = max(1, 128 // args.batch_size)

    for epoch in range(args.epochs):
        running_loss, correct = 0, 0
        mean_logits = torch.zeros(num_models)

        for batch, (X, y) in enumerate(tqdm(data_loader)):
            X, y = X.to(rank), y.to(rank)

            with torch.no_grad():
                outputs = [model(X) for model in model.models]
            pred = stack_model(torch.stack(outputs, dim=-1))
            loss = loss_fn(pred, y)
            mean_logits += torch.tensor([outs.mean() for outs in outputs])

            loss.backward()

            if (batch + 1) % accumulation_steps == 0 or batch + 1 == len(data_loader):
                optimiser.step()
                optimiser.zero_grad()

            # Sum correct predictions
            y_pred = torch.argmax(pred, dim=1)
            correct += (y_pred == y).sum().item()
            running_loss += loss.item()

        correct /= len(data_loader.dataset)
        running_loss /= len(data_loader)
        mean_logits /= len(data_loader)

        print(f"Accuracy: {(100 * correct):>0.2f}%, Avg loss: {running_loss:>5f}")

    # Update the json config with new weights
    stack_weights = stack_model.fc.weight
    print(stack_weights)
    config['weights'] = stack_weights.flatten().tolist()
    with open(args.ensembe, 'w') as json_file:
        json.dump(config, json_file)


if __name__ == '__main__':
    args = parser.parse_args()
    main(0, args)

"""
Simple stacked ensemble implementation
"""
import torch
import torch.nn as nn
import timm


class Ensemble(nn.Module):
    def __init__(self, config: dict, num_classes: int):
        super().__init__()
        models = config['models']
        names = config['names']

        self.models = [
            timm.create_model(model, checkpoint_path=f'models/{name}.pth', num_classes=num_classes)
            for model, name in zip(models, names)
        ]
        self.weights = torch.tensor(config['weights'], dtype=torch.float).unsqueeze(-1).unsqueeze(-1)

    def forward(self, inputs):
        outputs = [model(inputs) for model in self.models]
        return torch.mean(torch.stack(outputs) * self.weights, dim=0)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

        for model in self.models:
            model.to(*args, **kwargs)

        self.weights = self.weights.to(*args, **kwargs)

    def eval(self):
        for model in self.models:
            model.eval()

    def train(self, mode: bool = True):
        for model in self.models:
            model.train(mode)

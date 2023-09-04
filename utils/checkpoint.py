"""
Module to contain classes to training history
"""
import json

import torch


class History:
    """History allowing for multiple unique stages with dynamic metrics"""
    def __init__(self):
        self.history = dict()
        self.current_stage = None

    def set_stage(self, stage: str, metrics: list):
        """Sets the current stage to write metrics to"""
        self.history[stage] = {m: [] for m in metrics}
        self.current_stage = stage

    def save(self, *args):
        """Saves args for the current stage"""
        for key, arg in zip(self.history[self.current_stage].keys(), args):
            self.history[self.current_stage][key].append(arg)

    def export(self, name: str):
        with open(f'models/{name}_history.json', 'w') as json_file:
            json.dump(self.history, json_file)

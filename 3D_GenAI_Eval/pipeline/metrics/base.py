# base.py
from abc import ABC, abstractmethod

class BaseMetric(ABC):
    def __init__(self, mesh_gt, mesh_ai):
        self.mesh_gt = mesh_gt
        self.mesh_ai = mesh_ai

    @abstractmethod
    def compute(self):
        pass

    def get_class(self, score, thresholds: dict, reverse: bool = False):
        """
        Generic classification logic based on thresholds.

        :param score: metric score
        :param thresholds: dictionary of threshold values
        :param reverse: if True, lower score is better
        :return: classification string
        """
        levels = list(thresholds.items())

        if reverse:
            # Lower is better: check from best to worst
            for label, th in levels:
                if score <= th:
                    return label
        else:
            # Higher is better: check from best to worst
            for label, th in levels:
                if score >= th:
                    return label

        # If nothing matched, return worst class
        return levels[-1][0]

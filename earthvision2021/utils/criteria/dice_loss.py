import torch
from torch.nn.modules.loss import _WeightedLoss


class DiceLoss(_WeightedLoss):
    def forward(self, prediction, label, weight=1):
        prediction = torch.nn.functional.softmax(prediction, dim=1)
        return self._calculate_loss(prediction, label, weight)

    @staticmethod
    def _calculate_loss(prediction, label, weight):
        eps = 0.0001
        one_hot_label = prediction.detach() * 0
        one_hot_label.scatter_(1, label.unsqueeze(1), 1)

        intersection = prediction * one_hot_label
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = prediction + one_hot_label
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss = weight * (1 - (numerator / denominator))
        return loss.sum() / prediction.size(1)



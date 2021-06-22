import logging
logger = logging.getLogger(__name__)
import numpy as np
import torch


class PrecisionRecall:
    def __init__(self, phase, classes):
        self.phase = phase  # train or val
        self.classes = classes
        # row is prediction, column is gt
        self.confusion_matrix = np.zeros((self.classes, self.classes))
        # suppress scientific notation and print floats with no decimals
        np.set_printoptions(suppress=True, formatter={'float_kind':'{:.0f}'.format})

    def __call__(self, prediction, label):
        prediction = torch.flatten(prediction).tolist()
        label = torch.flatten(label).tolist()
        for p, l in zip(prediction, label):
            self.confusion_matrix[p, l] += 1

    def calculate(self, eps=1e-8):
        # line break to make confusion matrix more legible
        logger.debug('\n' + str(self.confusion_matrix))
        tp = np.diag(self.confusion_matrix)
        fp = np.sum(self.confusion_matrix, axis=0) - tp
        fn = np.sum(self.confusion_matrix, axis=1) - tp
        tn = []
        for i in range(self.confusion_matrix.shape[1]):
            tmp = np.delete(self.confusion_matrix, i, 0)  # delete ith row
            tmp = np.delete(tmp, i, 1)  # delete ith column
            tn.append(sum(sum(tmp)))
        tn = np.array(tn)

        precision = np.mean((tp + eps) / (tp + fp + eps))
        recall    = np.mean((tp + eps) / (tp + fn + eps))
        iou = np.mean((tp + eps) / (tp + fp + fn + eps))
        f1 = (2 * precision * recall) / (precision + recall)
        stats_dict = {'recall': recall, 'precision': precision, 'iou': iou, 'f1': f1}
        stats_dict = dict(('{}_{}'.format(self.phase, k), v) for k, v in
                stats_dict.items())
        return stats_dict



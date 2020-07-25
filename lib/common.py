from . import data
import torch.nn.functional as F
import numpy as np
import torch

class Validator:
    def __init__(self, net, writer):
        self.net = net
        self.writer = writer

    def test(self, test_set, step):
        self.net.eval()
        x, target_labels = data.batch_gen(test_set, test_set.shape[0])
        predicted = self.net(x)
        loss = F.cross_entropy(predicted, target_labels)

        # calculate the accuracy
        _, preticted_labels = torch.max(predicted, 1)
        accuracy = torch.sum(target_labels == preticted_labels).item() / len(target_labels) * 100

        self.writer.add_scalar("testing loss", loss.item(), step)
        self.writer.add_scalar("testing accuracy", accuracy, step)
        self.net.train()

        return accuracy, loss.item()

def cal_loss(predicted, target):
    loss = F.cross_entropy(predicted, target)
    return loss
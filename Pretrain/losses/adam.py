"""Losses for Adam
"""
import torch
import torch.nn as nn


class PurposivePrunerLoss(nn.Module):
    '''Purposive Pruner Loss'''
    def __init__(self,sim_threshold=0.8):
        super(PurposivePrunerLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.sim_threshold = sim_threshold

    def forward(self, output, target, cosine_similarities,args):
        '''Forward pass'''
        losses = []
        for i in range(output.shape[0]):
            label = torch.zeros([1],dtype=torch.long).cuda()
            o=output[i].cuda()
            c=cosine_similarities[i].cuda()
            mask=c<self.sim_threshold
            gt = torch.tensor([True], dtype=torch.bool).cuda()
            mask = torch.cat((gt, mask), 0).cuda()
            masked_output = torch.masked_select(o, mask)
            masked_output = torch.unsqueeze(masked_output, 0)
            loss=self.criterion(masked_output,label)
            losses.append(loss)

        losses = torch.stack(losses).mean()
        return losses

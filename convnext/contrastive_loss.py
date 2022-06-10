from sklearn.metrics import euclidean_distances
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self,margin=1.0) -> None:
        super(ContrastiveLoss,self).__init__()
        self.margin = margin
        self.eps = 0.0001
    
    def forward(self, x0: torch.Tensor,x1: torch.Tensor,label):
        # print('label size: ',label)

        # print(x0[0].pow(2).sum())
        # exit()
        euclidean_distance = (x0-x1).pow(2).sum(1)
        # euclidean_distance = euclidean_distance.sqrt()
        # print(euclidean_distance.size())
        # print('EUCLIDEAN: ',euclidean_distance)
        # op_res = (1-label) * (0.5)*torch.pow(euclidean_distance, 2) + (label) * (0.5)*torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        # print(op_res.size())
        # print(op_res)
        # print(torch.mean(op_res))
        
        loss_contrastive = torch.mean( (label)* (0.5)*torch.pow(euclidean_distance, 2) +
                                     (1-label)* (0.5)*torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        # print(loss_contrastive)
        return loss_contrastive
        # # Calculate the euclidean distance and calculate the contrastive loss
        # # print('out1: ',output1)
        # # print('out2: ',output2)
        # # print('diff: ', difference)
        # # print(difference.size())
        # # print('sqrd distance: ',squared_distance)
        # # print(squared_distance.size())
        # # print('euclidean_dist: ',euclidean_distance)

        # # euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

        # loss_contrastive = torch.mean((1-label) * (1/2)*squared_distance +
        #                             label*(1/2)*torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        # print("loss contrastive: ",loss_contrastive)
        # # exit()

        # return loss_contrastive
import torch

class ContrastiveLoss(nn.Module):
    def __init__(self,margin=1.0) -> None:
        super(ContrastiveLoss,self).__init__()
        self.margin = margin
    
    def forward(self, t1: torch.Tensor,t2: torch.Tensor,label):
        t12 = t2-t1
        euclidean = torch.sqrt(torch.sum(t12.pow(2)))
        margin = 1
        loss = (1-label)*(1/2)*(euclidean)+label*(1/2)*torch.pow(torch.max(0,margin-euclidean),2)
        return loss
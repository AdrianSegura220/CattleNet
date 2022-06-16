import torch.nn as nn
import torch

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchorExample: torch.Tensor, positiveExample: torch.Tensor, negativeExample: torch.Tensor) -> torch.Tensor:
        distancePositiveExample = self.euclidean(anchorExample, positiveExample)
        distanceNegativeExample = self.euclidean(anchorExample, negativeExample)
        losses = torch.relu(distancePositiveExample - distanceNegativeExample + self.margin)

        return losses.mean()

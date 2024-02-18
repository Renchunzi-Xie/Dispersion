import torch
from algs.base_alg import Base_alg
import torch.nn as nn
from common.distance import *

score_dict = {
    "dispersion": dispersion
}

class Dispersion(Base_alg):
    def __init__(self,val_loader, device, args):
        super(Dispersion, self).__init__(val_loader, device, args)
        # load the pre-trained model.
        features = list(self.base_model.children())[:-1]
        self.modelout = nn.Sequential(*features).to(self.device)
        classifier = list(self.base_model.children())[-1:]
        self.classifier = nn.Sequential(*classifier).to(self.device)

    def evaluate(self):
        self.base_model.eval()
        z = []
        y_hat = []
        for batch_idx, batch_data in enumerate(self.val_loader):
            inputs, labels = batch_data[0], batch_data[1]
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                hidden_values = self.modelout(inputs)
                hidden_values2 = hidden_values.view(labels.shape[0], -1)
                z.append(hidden_values2.detach())
                p = self.classifier(hidden_values2)
                prob, pseudo_labels = p.max(1)
                y_hat.append(pseudo_labels.detach())
        z = torch.cat(z)
        y_hat = torch.cat(y_hat)
        return score_dict[self.args["score"]](z.cpu(), y_hat.cpu())



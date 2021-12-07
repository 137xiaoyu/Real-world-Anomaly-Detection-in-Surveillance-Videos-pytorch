import torch.nn as nn


class Learner(nn.Module):
    def __init__(self, input_dim=2048, drop_p=0.6):
        super(Learner, self).__init__()
        self.drop_p = drop_p
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.drop_p),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(self.drop_p),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.classifier(x)

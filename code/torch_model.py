import torch

class FeatureExtractionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extraction = torch.nn.Sequential(
            torch.nn.Conv2d(3, 10, kernel_size=5),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(10, 30, kernel_size=5),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(30, 50, kernel_size=5),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(50, 80, kernel_size=5),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(80, 120, kernel_size=5),
        )

        self.head = torch.nn.Sequential(
            torch.nn.Linear(120, 60),
            torch.nn.ReLU(),
            torch.nn.Linear(60, 5),
            torch.nn.Softmax(dim=-1),
        )

    def forward(self, x):
        features = self.feature_extraction(x).mean(dim=[-1, -2])
        return features, self.head(features)

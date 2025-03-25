import torch.nn as nn

# Define the 1D CNN model
class Cnn1dModel(nn.Module):
    def __init__(self):
        super(Cnn1dModel, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=25, kernel_size=3),
            nn.BatchNorm1d(25),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=25, out_channels=50, kernel_size=3),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=50, out_channels=75, kernel_size=3),
            nn.BatchNorm1d(75),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.AdaptiveMaxPool1d(1), #Using Global Average Pooling to reduce the number of parameters
            nn.Flatten(),
            nn.Linear(in_features=75, out_features=512),

            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1),
            nn.Sigmoid()  # Sigmoid for binary classification (output between 0 and 1)
        )

    def forward(self, x):
        return self.model(x)
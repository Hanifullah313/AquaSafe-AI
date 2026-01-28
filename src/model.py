import torch
import torch.nn as nn

class WaterQualityNet(nn.Module):
    def __init__(self, input_features=9):
        super(WaterQualityNet, self).__init__()
        
        
        self.model = nn.Sequential(

            # --- Layer 1: Input Layer ---
            nn.Linear(input_features, 256),  # 9 inputs -> 256 neurons
            nn.ReLU(),                      # Activation
            nn.Dropout(0.1),  
                          # Randomly turn off 30% of neurons
            # --- Layer 1: Input Layer ---
            nn.Linear(256, 128),  # 256 inputs -> 128 neurons
            nn.ReLU(),                      # Activation
            nn.Dropout(0.1),                # Randomly turn off 30% of neurons
            
            #-- Layer 2: Hidden Layer ---
            nn.Linear(128, 64),              # 128 neurons -> 64 neurons
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # --- Layer 3: Hidden Layer ---
            nn.Linear(64, 32),               # 64 neurons -> 32 neurons
            nn.ReLU(),

            # --- Output Layer ---
            nn.Linear(32, 1),                # 32 neurons -> 1 score
            nn.Sigmoid()
            
                               
        )

    def forward(self, x):
        # We just pass the input 'x' to the sequential model
        return self.model(x)
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class WaterQualityDataset(Dataset):
    def __init__(self, features, labels):
        # We ensure inputs are tensors and correct types here
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1) # Make shape (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Returning features and label directly is often easier than a dict
        return self.X[idx], self.y[idx]


def get_dataloaders(csv_path, batch_size=32):
    """
    Loads data, cleans it, scales it, and returns PyTorch DataLoaders.
    """
    # 1. Load Data
    df = pd.read_csv(csv_path)
    
    X = df.drop('Potability', axis=1).values
    y = df['Potability'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 5. Train / Test Split (80% Train, 20% Val)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. Create Datasets
    train_ds = WaterQualityDataset(X_train, y_train)
    val_ds = WaterQualityDataset(X_val, y_val)
    
    # 7. Create Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scaler
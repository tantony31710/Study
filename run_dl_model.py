import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

# 1. Load preprocessed data
data = pd.read_csv('mental_health_cleaned.csv')

categorical_cols = ['Code']
numerical_cols = [
    'Schizophrenia disorders (share of population)',
    'Depressive disorders (share of population)',
    'Anxiety disorders (share of population)',
    'Bipolar disorders (share of population)',
    'Eating disorders (share of population)'
]
X = data[categorical_cols + numerical_cols]
min_year = data['Year'].min()
y = data['Year'].values - min_year 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', StandardScaler(), numerical_cols),
    ]
)

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Convert to Tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 2. Architecture
class RegressionMLP(nn.Module):
    def __init__(self, input_dim):
        super(RegressionMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

# 3. Training
model = RegressionMLP(input_dim=X_train_tensor.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

epochs = 100 # Reduced for test run
print('Training the model...')
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        rmse = torch.sqrt(loss).item()
        print(f'Epoch [{epoch+1}/{epochs}], Training RMSE: {rmse:.4f} years')

# 4. Export to ONNX
model.eval()
dummy_input = torch.randn(1, 210)
torch.onnx.export(
    model,
    dummy_input,
    "mental_health_nn_test.onnx",
    export_params=True,
    opset_version=18,
    input_names=['Mental_Health_Metrics'],
    output_names=['Predicted_Year_Offset']
)
print("Model compiled and saved successfully as mental_health_nn_test.onnx!")

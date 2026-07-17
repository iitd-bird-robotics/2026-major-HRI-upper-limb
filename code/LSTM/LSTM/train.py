import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from lstm import Phase_classifier  
from dataloader import PhaseDataset  

# === Config ===
data_path = 'C:/mtp/New folder/arm26-main/output_case_b.csv'   
model_save_path = 'C:/mtp/New folder/arm26-main/lstm_model.pth'
seq_length = 10
batch_size = 64
num_epochs = 50
input_size = 4
num_classes = 4

# === Load Dataset ===
data = pd.read_csv(data_path)
print(f"Initial shape: {data.shape}")

X_raw = data[["s_emg1", "s_emg2", "s_emg3", "s_emg4"]]
y_raw = data[["a_emg1", "a_emg2", "a_emg3", "a_emg4",
              "s_emg1", "s_emg2", "s_emg3", "s_emg4"]]

uncertainty = pd.DataFrame({
    "emg1_uncertainty": y_raw["a_emg1"] - y_raw["s_emg1"],
    "emg2_uncertainty": y_raw["a_emg2"] - y_raw["s_emg2"],
    "emg3_uncertainty": y_raw["a_emg3"] - y_raw["s_emg3"],
    "emg4_uncertainty": y_raw["a_emg4"] - y_raw["s_emg4"],
})

# uncertainty = data[["a_emg1", "a_emg2", "a_emg3", "a_emg4"]]

df = pd.concat([X_raw, uncertainty], axis=1).dropna()

X = df[["s_emg1", "s_emg2", "s_emg3", "s_emg4"]].values
y = df[["emg1_uncertainty", "emg2_uncertainty", "emg3_uncertainty", "emg4_uncertainty"]].values


dataset = PhaseDataset(data_path=df, seq_length=seq_length)
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# === Model Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")               
model = Phase_classifier(input_size=input_size, num_classes=num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Training ===
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device).float(), labels.to(device).long()
        # print(inputs.shape, labels.shape,"Input and output shape")  # Debugging line to check input shapes
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

# === Save Model ===
torch.save(model.state_dict(), model_save_path)

# === Evaluation ===
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device).float(), labels.to(device).long()
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

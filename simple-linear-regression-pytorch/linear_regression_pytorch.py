import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# Make sure the screenshots folder exists
os.makedirs('screenshots', exist_ok=True)

# 1. Generate synthetic data: y = 2x + 3 + noise
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100 data points between 0 and 10
y = 2 * X + 3 + np.random.randn(100, 1)  # Add some noise

# Convert to PyTorch tensors
X_train = torch.from_numpy(X.astype(np.float32))
y_train = torch.from_numpy(y.astype(np.float32))

# 2. Define the model
model = nn.Linear(1, 1)  # One input, one output

# 3. Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. Train the model
epochs = 300
for epoch in range(epochs):
    # Forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Print every 50 epochs to show improvement
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 5. Results
[w, b] = model.parameters()
final_loss = loss.item()
learned_weight = w.item()
learned_bias = b.item()
print(f'Learned weight: {learned_weight:.2f}, bias: {learned_bias:.2f}, Final loss: {final_loss:.4f}')

# 6. Plot and save the result, including learned weight, bias, and loss
predicted = model(X_train).detach().numpy()
plt.scatter(X, y, label='Original Data')
plt.plot(X, predicted, color='red', label='Fitted Line')
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression with PyTorch")

# Annotate with learned values and loss
plt.text(
    0.05, 0.95,
    f'Weight: {learned_weight:.2f}\nBias: {learned_bias:.2f}\nLoss: {final_loss:.4f}',
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7)
)

plt.savefig('screenshots/result_plot.png')
plt.show()

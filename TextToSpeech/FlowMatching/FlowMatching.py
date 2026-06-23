import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. Creating Clean Dummy Dataset
# ==========================================

seq_len = 100
time_steps = torch.linspace(0, 4 * math.pi, seq_len)
clean_data = torch.sin(time_steps)
pure_noise = torch.randn(seq_len)

# ==========================================
# 2. Flow Matching Network
# ==========================================

class FlowPredictor(nn.Module):
    def __init__(self):
        super(FlowPredictor, self).__init__()
        self.net = nn.Sequential(
            # FIX 1: Changed out_features from 128 to 256 to match the input of the next layer
            nn.Linear(seq_len + 1, 256),
            nn.ReLU(),
            nn.Linear (256, 256),
            nn.ReLU(),
            nn.Linear(256, seq_len)
        )
        
    def forward(self, noisy_x, t):
        t_tensor = torch.full((1,), t)
        input_tensor = torch.cat([noisy_x, t_tensor])
        return self.net(input_tensor)

model = FlowPredictor()
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

# ==========================================
# 3. Training the Flow Model
# ==========================================
print("Training the Flow Model...")
epochs = 1000

for epoch in range(epochs):
    optimizer.zero_grad()
    
    t = torch.rand(1).item()
      
    blended_x = (1 - t) * pure_noise + t * clean_data
    
    target_direction = clean_data - pure_noise
    
    predicted_direction = model(blended_x, t)
    
    loss = criterion(predicted_direction, target_direction)
    loss.backward()
    optimizer.step()
    
print("Training Complete!")

# ==========================================
# 4. Inference (Generating from scratch!)
# ==========================================
print("Generating a new sample from scratch...")

generated_signal = torch.randn(seq_len)

num_steps = 10
dt = 1.0 / num_steps

history = [generated_signal.clone().detach()]

with torch.no_grad():
    for step in range(num_steps):
        t = step * dt
        predicted_direction = model(generated_signal, t)
        generated_signal += predicted_direction * dt
        history.append(generated_signal.clone().detach())


# ==========================================
# 5. Visualization
# ==========================================
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        
axs[0].plot(clean_data.numpy(), color='green')
axs[0].set_title("Target Clean Signal")
axs[0].set_ylim([-2.5, 2.5])        

axs[1].plot(history[0].numpy(), color='red')
axs[1].set_title("Starting Pure Noise (t=0.0)")
axs[1].set_ylim([-2.5, 2.5])

axs[2].plot(history[-1].numpy(), color='blue')
axs[2].set_title(f"AI Generated Signal (t=1.0, {num_steps} steps)")
axs[2].set_ylim([-2.5, 2.5])

plt.tight_layout()
plt.show()
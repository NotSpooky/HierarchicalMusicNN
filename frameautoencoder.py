import torch

class Autoencoder(torch.nn.Module):
  def __init__(self, input_size, hidden_size):
    super(Autoencoder, self).__init__()
    self.encoder = torch.nn.Linear(input_size, hidden_size)
    self.activation = torch.nn.SELU()
    self.encoder2 = torch.nn.Linear(hidden_size, hidden_size)
    self.decoder = torch.nn.Linear(hidden_size, hidden_size)
    self.activation2 = torch.nn.SELU()
    self.decoder2 = torch.nn.Linear(hidden_size, input_size)
    self.activation3 = torch.nn.ReLU()
    self.decoder3 = torch.nn.Linear(input_size, input_size)
    
  def forward(self, x):
    # Use the square root of the spectrogram as input
    x = self.encoder(x)
    x = self.activation(x)
    x = self.encoder2(x)
    x = self.decoder(x)
    x = self.decoder2(x)
    x = self.activation2(x)
    x = self.decoder3(x)
    x = self.activation3(x)
    # Get the power of 3 of the reconstructed spectrogram
    x = torch.pow(x, 3)
    return x

def train(spectrogram, device):
  # Create an autoencoder using the provided device
  autoencoder = Autoencoder(1024, 512).to(device)

  # Define the loss function
  criterion = torch.nn.MSELoss()

  # Define the optimizer
  optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0005)

  # Train the autoencoder
  epochs = 30
  batch_size = 32
  for epoch in range(epochs):
    i = 0

    # Ceate the batches
    while i < spectrogram.shape[0]:
      batch = spectrogram[i:i+batch_size].to(device)
      i += batch_size

      # Forward pass
      output = autoencoder(batch)
      loss = criterion(output, batch)

      # Backward pass
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # Print each 32th batch
      if i % 32 == 0:
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, epochs, loss.item()))
      i += 1

    print('Epoch {}/{}, Loss: {}'.format(epoch+1, epochs, loss.item()))

  # Return the model and optimizer
  return autoencoder, optimizer

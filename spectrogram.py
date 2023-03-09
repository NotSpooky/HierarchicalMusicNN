import torch
import librosa
import numpy
import frameautoencoder
import os
import soundfile as sf
import sys

force_train = False
# Check if the --force_train argument is passed
if '--force_train' in sys.argv:
  force_train = True

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load an audio file
audio, sr = librosa.load('rigel.mp3', sr=44100)

# Convert the audio signal to a tensor
audio_tensor = torch.from_numpy(audio).unsqueeze(0)

# Apply the Short-Time Fourier Transform (STFT)
stft = torch.stft(audio_tensor, n_fft=1024, hop_length=512, window=torch.hann_window(1024), normalized=True, return_complex=True)
print (stft.shape) # torch.Size([1, 513, N])

complex_spectrogram = torch.cat([stft.real, stft.imag], dim=-1)
print("Complex spectrogram shape: ", complex_spectrogram.shape)

# spectrogram_db_original = librosa.amplitude_to_db(stft.squeeze().numpy())

real_spectrogram = torch.view_as_real(stft)

print("Real spectrogram shape: ", real_spectrogram.shape) # torch.Size([1, 513, N, 2])

# Change the spectrogram to be fit for the neural network
permuted_spectrogram = real_spectrogram.permute(0, 2, 3, 1)

print("Permuted spectrogram shape: ", permuted_spectrogram.shape) # torch.Size([1, N, 2, 513])

merged_spectrogram = permuted_spectrogram.reshape(1, -1, 1026)

print("Merged spectrogram shape: ", merged_spectrogram.shape) # torch.Size([1, N, 1026])

# Remove the last 2 values of each frame so that they're of size 1024
trimmed_spectrogram = merged_spectrogram[:, :, :-2]

# Check if the checkpoint exists
if os.path.isfile('checkpoint.pth') and not force_train:
  print("=> loading checkpoint")
  model = frameautoencoder.Autoencoder(1024, 512).to(device)
  checkpoint = torch.load('checkpoint.pth')
  model.load_state_dict(checkpoint['state_dict'])
  print("=> loaded checkpoint")
else:
  # frameautoencoder.train(trimmed_spectrogram, device)
  if force_train:
    print("=> (forced) training model")
  else:
    print("=> no checkpoint found")
  # Train the model
  model, optimizer = frameautoencoder.train(trimmed_spectrogram, device)
  # Save the model
  torch.save({
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    }, 'checkpoint.pth')

# Apply the model to the spectrogram
output = model(trimmed_spectrogram.to(device))

# Print the output shape
print("Output shape: ", output.shape) # torch.Size([1, N, 1024])

# Append 0s to have the same size as the original spectrogram [1, N, 1026]
decoded_spectrogram = torch.cat((output, torch.zeros(1, output.shape[1], 2).to(device)), dim=-1)

print("Decoded shape: ", decoded_spectrogram.shape) # torch.Size([1, N, 1026])

reversed_merging = merged_spectrogram.reshape(1, -1, 2, 513)
print("Reversed merged spectrogram shape: ", reversed_merging.shape) # torch.Size([1, N, 2, 513])

reverse_permutation = reversed_merging.permute(0, 3, 1, 2)
print("Reverse permutation shape: ", reverse_permutation.shape) # torch.Size([1, 513, N, 2])

reverse_to_complex = torch.view_as_complex(reverse_permutation.contiguous())
print("Reverse to complex shape: ", reverse_to_complex.shape) # torch.Size([1, 513, N])

# # Check the reversal
# print (torch.allclose(reverse_permutation, real_spectrogram))

audio_reconstructed = torch.istft(
  reverse_to_complex,
  n_fft=1024,
  hop_length=512,
  window=torch.hann_window(1024),
  normalized=True
)

# Convert the reconstructed audio signal to a NumPy array
audio_reconstructed = audio_reconstructed.squeeze().numpy()

# Generate a wav file from the reconstructed audio signal
sf.write('rigel_reconstructed.wav', audio_reconstructed, sr)

print("Saved rigel_reconstructed.wav")

"""


# Convert the spectrogram to a decibel (dB) scale

# Plot the spectrogram
import matplotlib.pyplot as plt
# plt.imshow(spectrogram_db, aspect='auto', origin='lower', cmap='magma')
# plt.colorbar()
# plt.xlabel('Time (frames)')
# plt.ylabel('Frequency (Hz)')
# plt.show()

# Transpose the spectrogram to match the input size of the autoencoder
spectrogram = spectrogram.transpose(1, 2)

# Remove the last element of the frames to be of length 512
spectrogram = spectrogram[:, :, :-2]

frameautoencoder.train(spectrogram)

# Pass the spectrogram through the autoencoder
decoded_spectrogram = []
for frame in spectrogram[0]:
  decoded = autoencoder.forward(frame.to(device))
  # Convert the tensor to a numpy array
  decoded = decoded.detach().cpu().numpy()
  decoded_spectrogram.append(decoded)

# Convert the encoded spectrogram to a tensor
decoded_spectrogram = torch.tensor(numpy.array(decoded_spectrogram))

# Add a [0,0] to the end of each frame
decoded_spectrogram = torch.cat((decoded_spectrogram, torch.zeros(decoded_spectrogram.shape[0], 2)), dim=1)

print (decoded_spectrogram.shape) # torch.Size([N, 1026])

# Unsqueeze the encoded spectrogram to match the input size of the inverse STFT
decoded_spectrogram = decoded_spectrogram.unsqueeze(0)

# Attach the phase to the encoded spectrogram so it has 2 dimensions for the reversal
decoded_spectrogram = decoded_spectrogram.transpose(1, 2)

# Revert to [513, N, 2]
decoded_spectrogram = decoded_spectrogram.reshape(513, -1, 2)

decoded_spectrogram = torch.view_as_complex(decoded_spectrogram)

# Reverse the STFT to get the original audio signal
audio_reconstructed = torch.istft(
  decoded_spectrogram,
  n_fft=1024,
  hop_length=512,
  window=torch.hann_window(1024),
  normalized=True
)

# Convert the reconstructed audio signal to a NumPy array
audio_reconstructed = audio_reconstructed.squeeze().numpy()

# Generate a wav file from the reconstructed audio signal
import soundfile as sf
sf.write('rigel_reconstructed.wav', audio_reconstructed, sr)

import matplotlib.pyplot as plt

# Create a figure with two subplots arranged side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot the original audio signal on the first subplot
axs[0].imshow(spectrogram_db_original, aspect='auto', origin='lower', cmap='magma')
axs[0].set_xlabel('Time (frames)')
axs[0].set_ylabel('Frequency (Hz)')
axs[0].set_title('Original')

# Plot the reconstructed audio signal on the second subplot
spectrogram_db_reconstructed = librosa.amplitude_to_db(decoded_spectrogram.squeeze().detach().cpu().numpy(), ref=numpy.max)
axs[1].imshow(spectrogram_db_reconstructed, aspect='auto', origin='lower', cmap='magma')
axs[1].set_xlabel('Time (frames)')
axs[1].set_ylabel('Frequency (Hz)')
axs[1].set_title('Reconstructed')



# Add a colorbar for both subplots
cbar = fig.colorbar(axs[0].imshow(spectrogram_db_original, aspect='auto', origin='lower', cmap='magma'), ax=axs, orientation='vertical')

# Show the plot
plt.show()
"""

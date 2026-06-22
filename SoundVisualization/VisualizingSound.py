import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import math


SAMPLE_RATE = 16000  
DURATION = 2         
total_samples = SAMPLE_RATE * DURATION


time_tensor = torch.linspace(0, DURATION, total_samples)
low_pitch = torch.sin(2 * math.pi * 400 * time_tensor)
high_pitch = torch.sin(2 * math.pi * 1000 * time_tensor)


raw_waveform = low_pitch + high_pitch + (torch.randn(total_samples) * 0.1)

raw_waveform = raw_waveform.unsqueeze(0) 

print(f"Waveform Shape: {raw_waveform.shape} -> [Channels: 1, Data Points: 32000]")


mel_spectrogram_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,          
    hop_length=256,      
    n_mels=80            
)


mel_spec = mel_spectrogram_transform(raw_waveform)


amplitude_to_db_transform = T.AmplitudeToDB(stype='power', top_db=80)
mel_spec_db = amplitude_to_db_transform(mel_spec)

print(f"Spectrogram Shape: {mel_spec_db.shape} -> [Channels: 1, Mel-Bands: 80, Time-Frames: 126]")


print("Generating Audio Visualizations...")

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

axs[0].plot(raw_waveform[0, :1000].numpy(), color='blue')
axs[0].set_title("Raw Audio Waveform (First 1000 samples)")
axs[0].set_ylabel("Amplitude (Loudness)")
axs[0].set_xlabel("Time (Samples)")


im = axs[1].imshow(mel_spec_db[0].numpy(), cmap='magma', origin='lower', aspect='auto')
axs[1].set_title("Mel-Spectrogram (What the AI actually sees)")
axs[1].set_ylabel("Frequency (80 Mel-Bands)")
axs[1].set_xlabel("Time (Frames)")


fig.colorbar(im, ax=axs[1], format="%+2.0f dB")

plt.tight_layout()
plt.show()
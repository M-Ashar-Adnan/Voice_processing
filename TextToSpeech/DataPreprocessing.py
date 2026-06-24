import os
import re
import math
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
import soundfile as sf

# Making new directories for raw and processed data
os.makedirs('./raw_dataset/audio', exist_ok=True)
os.makedirs('./processed_dataset/audio', exist_ok=True)

print("📥 Fetching real speech data sample...")

# We load from librosa's cache, but save to our local raw dataset folder
librosa_cache_path = librosa.ex('libri1')
raw_audio_path = './raw_dataset/audio/sample_001.wav'

y, sr = librosa.load(librosa_cache_path, sr=44100) 

stereo_test_signal = np.vstack([y, y])
sf.write(raw_audio_path, stereo_test_signal.T, sr)
raw_transcript = "In 2026, he read 5 books! شاہکار "
print(f"Loaded Raw Transcript: {raw_transcript}\n" + "-"*50)


# =====================================================================
# Core Audio Preprocessing Functions
# =====================================================================
class AudioProcessor:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
        
    def process_file(self, input_path, output_path):
        try:
            waveform, sr = torchaudio.load(input_path)
        except Exception as e:
            print(f"torchaudio failed to load ({e}). Using librosa fallback...")
            try:
                y_np, sr = librosa.load(input_path, sr=None, mono=False)
                if y_np.ndim == 1:
                    y_np = np.expand_dims(y_np, axis=0)
                waveform = torch.tensor(y_np, dtype=torch.float32)
            except Exception as fallback_e:
                print(f"Corrupted File! Both torchaudio and librosa failed: {fallback_e}")
                return None
        
        print(f"Processing raw audio: shape {waveform.shape}, Original SR: {sr}Hz")

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            print(f"Converted to mono: shape {waveform.shape}")

        if sr != self.target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)
            print(f"Resampled Consistency: Converted from {sr}Hz to {self.target_sr}Hz, New shape: {waveform.shape}")

        audio_np = waveform.squeeze().numpy()

        trimmed_audio, index = librosa.effects.trim(audio_np, top_db=20, frame_length=1024, hop_length=256)
        print(f"Trim Excessive Silence: Removed {len(audio_np) - len(trimmed_audio)} samples of silence, New shape: {trimmed_audio.shape}")

        max_val = np.max(np.abs(trimmed_audio))
        if max_val > 0:
            normalized_audio = trimmed_audio / max_val
            print("Normalize Volume: Peak balance to 1.0 (Prevent Clipping/Distortion).")
        else:
            normalized_audio = trimmed_audio

        try:
            torchaudio.save(output_path, torch.tensor(normalized_audio).unsqueeze(0), self.target_sr)
        except Exception as e:
            print("torchaudio save failed. Using soundfile fallback...")
            sf.write(output_path, normalized_audio, self.target_sr)
            
        return normalized_audio
    
    
# =====================================================================
# Transcript Preprocessing Function
# =====================================================================
class TextPreprocessor:
    def __init__(self):
        self.num_map = {'0':'zero', '1':'one', '2':'two', '3':'three', '4':'four', '5':'five', '6':'six', '7':'seven', '8':'eight', '9':'nine', '2026':'twenty twenty six'}
            
    def clean_text(self, text):
        cleaned = text.lower()

        for num_word, text_word in self.num_map.items():
            cleaned = cleaned.replace(num_word, text_word)
        print("Expand Numbers/Dates: Applied Machine Mappings.")

        cleaned = re.sub(re.compile(r'[^\w\s\u0600-\u06FF]'), '', cleaned)
        print ("Remove Weird symbols: Cleared specialized punctuation markings.")

        cleaned = " ".join(cleaned.split())
        return cleaned
        

# =====================================================================
# Execution & Metric Calculations
# =====================================================================
audio_proc = AudioProcessor(target_sr=16000)
text_proc = TextPreprocessor()

processed_audio = audio_proc.process_file(raw_audio_path, "./processed_dataset/audio/sample_001.wav")
processed_text = text_proc.clean_text(raw_transcript)
        
print("\n" + "="*50 + "\nDataset statistics and quality checks \n" + "="*50)

if processed_audio is not None:
    duration_secs = len(processed_audio) / 16000
    rms_energy = np.sqrt(np.mean(processed_audio**2))
    noise_floor = np.percentile(np.abs(processed_audio), 10)
            
    print(f"• Total Computed Sample Duration : {duration_secs:.2f} seconds")
    print(f"• Avg RMS Energy Level           : {rms_energy:.4f} (Volume Check)")
    print(f"• Calculated Background Noise    : {noise_floor:.5f} (Noise Level Check)")
    print(f"• Clipping Status Check          : {'CLIPPING DETECTED' if np.max(np.abs(processed_audio)) > 1.0 else 'Safe (No Clipping)'}")
    print(f"• Cleaned Pipeline Transcript    : '{processed_text}'")
else:
    print("Cannot calculate dataset statistics because the audio processing failed.")
import re
import os
import inflect
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, Audio
from datasets import load_dataset
import numpy as np
import soundfile as sf
from torch.nn.utils.rnn import pad_sequence


os.makedirs('./raw_dataset/audio', exist_ok=True)

print("Connecting to Hugging Face to stream LJSpeech (Parquet Mirror)...")

dataset = load_dataset("MikhailT/lj-speech", split="full", streaming=True)

dataset = dataset.cast_column("audio", Audio(decode=False))

metadata_lines = []

print('Saving 20 audio files from LJSpeech dataset...')
for i, item in enumerate(dataset):
    if i >= 20:
        break
    

    audio_bytes = item["audio"]["bytes"]
    

    text = item.get("normalized_text") or item.get("text", "")
    
    file_name = f"sample_{i:03d}.wav"
    file_path = f"./raw_dataset/audio/{file_name}"
    
    with open(file_path, "wb") as f:
        f.write(audio_bytes)
    
    metadata_lines.append(f"{file_name}|{text}")
    print(f"✅ Saved {file_name} with text: {text[:40]}...")
    
with open("./raw_dataset/metadata.csv", "w", encoding="utf-8") as f:
    f.write("\n".join(metadata_lines))
    
    
class audioProcessor:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
        self.mel_transoform = T.MelSpectrogram(
            sample_rate = target_sr,
            n_fft=1024,
            hop_length=256,
            n_mels=80
        )
        self.db_transform = T.AmplitudeToDB()
        
    def process(self, file_path):
        waveform, sr = torchaudio.load(file_path)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if sr != self.target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)
            
        max_val = waveform.abs().max()
        if(max_val >0):
            waveform = waveform / max_val
            
        mel_spec = self.mel_transoform(waveform)
        mel_spec_db = self.db_transform(mel_spec)
        
        return mel_spec_db.squeeze(0)
    
    
class TextPreprocessor:
    def __init__(self):
        self.inflect = inflect.engine()
        self.abbreviations = {
            r'\bMr\.\b': 'mister',
            r'\bMrs\.\b': 'misess',
            r'\bDr\.\b': 'doctor',
            r'\bNo\.\b': 'number',
            r'\bSt\.\b': 'saint',
            r'\bCo\.\b': 'company',
            r'\bJr\.\b': 'junior',
            r'\bMaj\.\b': 'major',
            r'\bGen\.\b': 'general',
            r'\bDrs\.\b': 'doctors',
            r'\bRev\.\b': 'reverend',
            r'\bHon\.\b': 'honorable',
            r'\bSgt\.\b': 'sergeant',
            r'\bCapt\.\b': 'captain',
            r'\bEsq\.\b': 'esquire',
            r'\bLtd\.\b': 'limited',
            r'\bCol\.\b': 'colonel',
            r'\bFt\.\b': 'fort'
        }
        
        self.vocab = "abcdefghijklmnopqrstuvwxyz'.?,-!"
        self.char_to_int = {char: i+1 for i, char in enumerate(self.vocab)}
        
        
        def expand_numbers(self, text):
            words = text.split()
            for i, word in enumerate(words):
                clean_word = re.sub(r'[,.]','', word)
                if clean_word.isdigit():
                    spelled_out = self.inflect_engine.number_towords(clean_word)
                    words[i] = spelled_out.replace('-', ' ')
            return " ".join(words)
        
        def expand_abbreviations(self, text):
            for regex, replacement in self.abbreviations.items():
                text = re.sub(regex, replacement, text, flags=re.IGNORECASE)
            return text
        def clean_and_tokenized(self, text):
            text = text.lower()
            text = self.expand_numbers(text)
            text = self.expand_abbreviations(text)
            text = re.sub(r'[^a-zA-Z0-9\s.,?!\'-]', '', text)
            tokens = [self.char_to_int[char] for char in text if char in self.char_to_int]
            return tokens
        
if __name__ == "__main__":
    processor = TextPreprocessor()
    test_sentence = "Mr. Smith bought 15 apples from St. John's market in 2024!"
    
    cleaned_tensor = processor.clean_and_tokenized(test_sentence)
    
print(f"Original: {test_sentence}")
print(f"Tokens:   {cleaned_tensor}")
    
#PyTorch DataSet:
class LJSpeechDataset(Dataset):
    def __init__(self, metadata_lines, audio_dir):
        self.audio_dir = audio_dir
        self.data_pairs = []
        
        self.audio_processor = audioProcessor()
        self.text_processor = TextPreprocessor()
        
        print(f"Parsing meta file from: {metadata_lines}")
        
        with open(metadata_lines, "r", encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts)==2:
                    self.data_pairs.append((parts[0], parts[1]))
                    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        file_name, raw_text = self.data_pairs[idx]
        file_path = os.path.join(self.audio_dir, file_name)
        audio_tensor = self.audio_processor.process(file_path)
        text_tensor = self.text_processor.clean_and_tokenize(raw_text)
        
        return audio_tensor, text_tensor
        
        
#The Collate Function (Dynamic Padding Engine)
def collate_fn(batch):
    audios, text = zip(*batch)
    audios_transposed = [a.transpose(0, 1) for a in audios]
    padded_audios = pad_sequence(audios_transposed, batch_first=True, padding_value=0.0)
    padded_audios = padded_audios.transpose(1, 2).unsqueeze(1)
    
    padded_texts = pad_sequence(texts, batch_fist=True, padding_value=0)
    
    return padded_audios, padded_texts


#Execution & Verification Testing
if __name__ == "__main__":
    print('---pipeline Testing---')
    
    dataset = LJSpeechDataset(
        metadata_lines="./raw_dataset/metadata.csv",
        audio_dir="./raw_dataset/audio"
    )
    
    print("Datasetloaded successfully!")
    
    dataloader = DataLoader(
        dataset,
        batch_size =4,
        shuffle=True,
        collate_fn=collate_fn   
    )
    print("\nSimulating data stream iteration (Fetching a batch)...")
    
    for batch_idx, (audio_batch, text_batch) in enumerate(dataloader):
        print(f"\n📦 --- SUCCESS: EXTRACTED BATCH {batch_idx + 1} ---")
        print(f"• Audio Batch Tensor Shape : {audio_batch.shape}")
        print(f"  └─ Dimensions guide      : [Batch Size, Channel, Mel-Bands, Max Time-Frames]")
        print(f"• Text Batch Tensor Shape  : {text_batch.shape}")
        print(f"  └─ Dimensions guide      : [Batch Size, Max Tokenized Sequence Length]")
        
        print(f"• Sample Text Vector tokens: {text_batch[0][:15]}...")
        
        break
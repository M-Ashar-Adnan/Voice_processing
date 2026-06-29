import os
from datasets import load_dataset, Audio

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
    
print("-" * 50)
print(f"🎉 Success! You now have a mini-dataset.")
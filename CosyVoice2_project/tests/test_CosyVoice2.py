from pathlib import Path
import sys
import wave


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from cosyvoice2_pipeline import DemoConfig, run_demo, split_text_into_chunks


def test_text_is_split_into_readable_chunks() -> None:
	text = "Hello there. This is a tiny CosyVoice 2 style demo for learning."

	chunks = split_text_into_chunks(text, characters_per_chunk=20)

	assert chunks
	assert chunks[0] == "Hello there."
	assert all(chunk for chunk in chunks)


def test_demo_pipeline_writes_a_valid_wav(tmp_path) -> None:
	output_path = tmp_path / "demo.wav"
	config = DemoConfig(sample_rate=16_000)
	report = run_demo("Hello world. This demo writes audio.", str(output_path), config)

	assert report.output_path == output_path.resolve()
	assert output_path.exists()

	with wave.open(str(output_path), "rb") as wav_file:
		assert wav_file.getnchannels() == 1
		assert wav_file.getsampwidth() == 2
		assert wav_file.getframerate() == 16_000
		assert wav_file.getnframes() > 0

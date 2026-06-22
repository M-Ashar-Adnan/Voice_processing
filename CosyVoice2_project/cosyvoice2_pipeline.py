
from __future__ import annotations

from dataclasses import dataclass
import argparse
import math
import re
import struct
import wave
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class DemoConfig:
    """Small knobs that control the demo synthesis behavior."""

    sample_rate: int = 22_050
    characters_per_chunk: int = 42
    seconds_per_token: float = 0.045
    minimum_chunk_seconds: float = 0.20
    silence_between_chunks_seconds: float = 0.05
    base_frequency_hz: float = 170.0
    pitch_step_hz: float = 14.0
    amplitude: float = 0.35


@dataclass(frozen=True)
class TextChunk:
    """One chunk of text passed through the demo TTS pipeline."""

    index: int
    text: str
    token_count: int
    duration_seconds: float
    pitch_hz: float


@dataclass(frozen=True)
class DemoReport:
    """A simple summary of what the pipeline produced."""

    input_source: str
    raw_text: str
    normalized_text: str
    chunks: List[TextChunk]
    output_path: Path


def load_text(source: str) -> str:
    """Load text from a file path or treat the input as direct text."""

    candidate = Path(source)
    if candidate.exists() and candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    return source


def normalize_text(text: str) -> str:
    """Clean up whitespace without changing the words themselves."""

    return re.sub(r"\s+", " ", text.strip())


def split_text_into_chunks(text: str, characters_per_chunk: int) -> List[str]:
    """Split text into small speech-sized chunks."""

    if not text:
        return []

    sentence_like_parts = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []

    for sentence in sentence_like_parts:
        sentence = sentence.strip()
        if not sentence:
            continue

        words = sentence.split()
        current_words: List[str] = []

        for word in words:
            current_words.append(word)
            joined = " ".join(current_words)

            # If the current chunk is long enough, finalize it and start again.
            if len(joined) >= characters_per_chunk:
                chunks.append(joined)
                current_words = []

        if current_words:
            chunks.append(" ".join(current_words))

    return chunks


def tokenize_words(text: str) -> List[str]:
    """Extract a tiny set of word-like tokens."""

    return re.findall(r"[A-Za-z0-9']+", text)


def build_text_chunks(text: str, config: DemoConfig) -> List[TextChunk]:
    """Convert text into chunk objects with fake acoustic metadata."""

    raw_chunks = split_text_into_chunks(text, config.characters_per_chunk)
    built_chunks: List[TextChunk] = []

    for index, chunk_text in enumerate(raw_chunks):
        token_count = max(1, len(tokenize_words(chunk_text)))

        duration_seconds = max(
            config.minimum_chunk_seconds,
            token_count * config.seconds_per_token,
        )

        pitch_hz = config.base_frequency_hz + (index % 5) * config.pitch_step_hz

        built_chunks.append(
            TextChunk(
                index=index,
                text=chunk_text,
                token_count=token_count,
                duration_seconds=duration_seconds,
                pitch_hz=pitch_hz,
            )
        )

    return built_chunks


def _tone_samples(
    frequency_hz: float,
    duration_seconds: float,
    sample_rate: int,
    amplitude: float,
) -> bytes:
    """Generate 16-bit mono PCM samples for one chunk."""

    total_frames = max(1, int(sample_rate * duration_seconds))
    fade_frames = max(1, min(total_frames // 8, int(sample_rate * 0.02)))
    pcm_frames = bytearray()

    for frame_index in range(total_frames):
        time_seconds = frame_index / sample_rate
        raw_sample = math.sin(2.0 * math.pi * frequency_hz * time_seconds)

        fade_in = min(1.0, frame_index / fade_frames)
        fade_out = min(1.0, (total_frames - frame_index - 1) / fade_frames)
        envelope = min(fade_in, fade_out)

        sample_value = int(32767 * amplitude * envelope * raw_sample)
        pcm_frames.extend(struct.pack("<h", sample_value))

    return bytes(pcm_frames)


def synthesize_audio(chunks: Sequence[TextChunk], config: DemoConfig) -> bytes:
    """Convert chunk metadata into a complete PCM audio buffer."""

    audio_buffer = bytearray()
    pause_frames = int(config.sample_rate * config.silence_between_chunks_seconds)
    pause_bytes = struct.pack("<h", 0) * pause_frames

    for index, chunk in enumerate(chunks):
        audio_buffer.extend(
            _tone_samples(
                frequency_hz=chunk.pitch_hz,
                duration_seconds=chunk.duration_seconds,
                sample_rate=config.sample_rate,
                amplitude=config.amplitude,
            )
        )

        if index < len(chunks) - 1:
            audio_buffer.extend(pause_bytes)

    return bytes(audio_buffer)


def write_wav_file(audio_bytes: bytes, output_path: Path, sample_rate: int) -> Path:
    """Write raw PCM bytes into a standard WAV container."""

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)

    return output_path


def run_demo(source: str, output_path: str, config: Optional[DemoConfig] = None) -> DemoReport:
    """Run the whole demo pipeline from text input to WAV output."""

    if config is None:
        config = DemoConfig()

    raw_text = load_text(source)
    normalized_text = normalize_text(raw_text)
    chunks = build_text_chunks(normalized_text, config)
    audio_bytes = synthesize_audio(chunks, config)
    final_output = write_wav_file(audio_bytes, Path(output_path), config.sample_rate)

    return DemoReport(
        input_source=source,
        raw_text=raw_text,
        normalized_text=normalized_text,
        chunks=chunks,
        output_path=final_output,
    )


def format_report(report: DemoReport) -> str:
    """Create a human-readable summary of the demo pipeline."""

    lines = [
        "CosyVoice 2 educational demo",
        f"Input source: {report.input_source}",
        f"Normalized text: {report.normalized_text}",
        f"Chunks generated: {len(report.chunks)}",
        f"Output WAV: {report.output_path}",
        "",
        "How the data flows:",
        "1. Text is loaded from a string or file.",
        "2. The text is normalized and split into smaller chunks.",
        "3. Each chunk gets a duration and pitch plan.",
        "4. The demo vocoder turns that plan into PCM samples.",
        "5. The samples are stored in a WAV file.",
    ]

    if report.chunks:
        lines.append("")
        lines.append("Chunk details:")
        for chunk in report.chunks:
            lines.append(
                f"- chunk {chunk.index}: {chunk.text!r} "
                f"({chunk.token_count} tokens, {chunk.duration_seconds:.2f}s, {chunk.pitch_hz:.1f}Hz)"
            )

    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the demo."""

    parser = argparse.ArgumentParser(
        description="Educational CosyVoice 2 style text-to-speech demo",
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=(
            "CosyVoice 2 is a text to speech system. This demo shows the data flow "
            "with comments and a simple synthesized output."
        ),
        help="Text to speak, or a path to a UTF-8 text file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="cosyvoice2_demo.wav",
        help="Where to write the generated WAV file.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DemoConfig.sample_rate,
        help="Audio sample rate used for the generated WAV file.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""

    parser = build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = DemoConfig(sample_rate=args.sample_rate)
    report = run_demo(args.source, args.output, config)

    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
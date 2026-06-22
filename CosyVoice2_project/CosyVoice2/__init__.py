"""Package exports for the educational CosyVoice 2 demo."""

from cosyvoice2_pipeline import (
	DemoConfig,
	DemoReport,
	TextChunk,
	build_argument_parser,
	build_text_chunks,
	format_report,
	load_text,
	main,
	normalize_text,
	run_demo,
	split_text_into_chunks,
	synthesize_audio,
	tokenize_words,
	write_wav_file,
)

import pyttsx3


def speak(text: str, rate: int = 100, volume: float = 1.0, voice_index: int = 0) -> None:
    """Convert text to speech and play it through the speakers.

    Args:
        text: The text to speak.
        rate: Words per minute (default 175).
        volume: Volume from 0.0 to 1.0 (default 1.0).
        voice_index: Which installed voice to use (default 0).
    """
    engine = pyttsx3.init()

    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)

    voices = engine.getProperty("voices")
    if voices:
        engine.setProperty("voice", voices[voice_index % len(voices)].id)

    engine.say(text)
    engine.runAndWait()
    engine.stop()


def save_to_file(text: str, filename: str = "output.mp3") -> None:
    """Convert text to speech and save it as an audio file instead of playing it."""
    engine = pyttsx3.init()
    engine.save_to_file(text, filename)
    engine.runAndWait()
    engine.stop()


def list_voices() -> None:
    """Print all voices available on this system."""
    engine = pyttsx3.init()
    for i, voice in enumerate(engine.getProperty("voices")):
        print(f"[{i}] {voice.name} ({voice.languages or voice.id})")


if __name__ == "__main__":
    print("Basic Text-to-Speech")
    print("Type a sentence and press Enter to hear it.")
    print("Commands: ':voices' to list voices, ':quit' to exit.\n")

    while True:
        user_input = input("> ").strip()

        if user_input == ":quit":
            break
        elif user_input == ":voices":
            list_voices()
        elif user_input:
            speak(user_input)
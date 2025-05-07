import pyaudio
import wave
import argparse
import os


def record_audio(
    output_filename,
    record_seconds,
    channels=1,
    rate=44100,
    chunk=1024,
    format=pyaudio.paInt16,
):
    audio = pyaudio.PyAudio()

    print("Recording audio... Press Ctrl+C to abort.")

    stream = audio.open(
        format=format,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=chunk,
    )
    frames = []

    try:
        for _ in range(0, int(rate / chunk * record_seconds)):
            data = stream.read(chunk)
            frames.append(data)
    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(output_filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b"".join(frames))
    wf.close()
    print(f"Audio saved to: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record audio from your HyperCast mic."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output WAV filename (default: output.wav)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Duration in seconds to record (default: 5)",
    )
    args = parser.parse_args()

    record_audio(args.output, args.duration)

    print("Playing back audio...")
    try:
        output = args.output
    except Exception:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        output = os.path.abspath(
            os.path.join(this_dir, "..", "..", "..", "output.wav")
        )
    wf = wave.open(output, "rb")
    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=audio.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True,
    )

    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    stream.stop_stream()
    stream.close()
    audio.terminate()
    wf.close()

    print("Playback finished.")

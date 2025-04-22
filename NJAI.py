import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import ollama

# Load Whisper model
model = whisper.load_model("tiny")

# Record your voice
fs = 44100  # Sampling rate
seconds = 5  # Duration of recording

print("🎙️ Start speaking now...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()
audio_path = 'test_audio.wav'
write(audio_path, fs, recording)

# Transcribe using Whisper
result = model.transcribe(audio_path)
user_text = result['text'].strip()

print("🗣️ You said:", user_text)

# Send to Ollama
response = ollama.chat(
    model='phi3:mini',
    messages=[{'role': 'user', 'content': user_text}]
)

# Print LLM response
print("\n🤖 Ollama:")
print(response['message']['content'])


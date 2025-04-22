import whisper
import ollama

# Load small model suitable for Jetson Orin
model = whisper.load_model("tiny")

# Record audio (5 sec example)
import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sampling rate
seconds = 5  # Duration
print("Start speaking now...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
write('test_audio.wav', fs, recording)

# Transcribe the audio file
result = model.transcribe('test_audio.wav')
print("You said:", result['text'])


try:
    user_input = result['text']

    response = ollama.chat(
        model='phi3:mini',
        messages=[
            {'role': 'system', 'content': 'You are an assistant who responds briefly, no more than 10 words.'},
            {'role': 'user', 'content': user_input}]
    )

    print("\n🤖 Ollama (phi3:mini):")
    print(response['message']['content'])

except Exception as e:
    print("❌ Error:", e)

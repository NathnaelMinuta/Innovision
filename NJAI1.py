import whisper
import ollama
import sounddevice as sd
from scipy.io.wavfile import write
from TTS.api import TTS
import subprocess

def voice_control():
	# Load Whisper and Coqui TTS models
	whisper_model = whisper.load_model("tiny")
	tts = TTS("tts_models/en/ljspeech/vits")

	# Record audio
	fs = 44100
	seconds = 5
	print("🎙️ Start speaking now...")
	recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
	sd.wait()
	write('test_audio.wav', fs, recording)

	# Transcribe
	result = whisper_model.transcribe('test_audio.wav')
	user_input = result['text'].strip()
	print("🗣️ You said:", user_input)

	# Ollama chat + Coqui TTS
	try:
	    response = ollama.chat(
		model='phi3:mini',
		messages=[
		    {'role': 'system', 'content': 'You are an assistant who responds briefly, no more than 10 words.'},
		    {'role': 'user', 'content': user_input}
		]
	    )

	    reply = response['message']['content'].strip()
	    print("\n🤖 Ollama (phi3:mini):", reply)

	    # Coqui speaks it
	    tts.tts_to_file(text=reply, file_path="response.wav")
	    subprocess.run(["aplay", "response.wav"])  # For Ubuntu/Jetson Orin

	except Exception as e:
	    print("❌ Error:", e)
	    



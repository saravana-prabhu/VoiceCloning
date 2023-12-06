
# Import the necessary libraries
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa 
from openai import OpenAI
from TTS.api import TTS
from pydub import AudioSegment

# Initialize XTTS and OpenAI
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
client = OpenAI(api_key="YOUR_API_KEY")

# Initialize file path
train_audio = "/home/saravana/VoiceClone/Git_Project/Train/dhoni_hindi.wav"
input_audio = "/home/saravana/VoiceClone/Git_Project/Input/dhoni_eng_1_min.wav"
output_audio_path = "/home/saravana/VoiceClone/Git_Project/Output/dhoni_hindi_output.wav"


# Function to concatenate audio files
def concatenate_audio(files, output_file):
    concatenated_audio = AudioSegment.silent(duration=0)

    for file in files:
        audio_segment = AudioSegment.from_file(file)
        concatenated_audio += audio_segment

    concatenated_audio.export(output_file, format="wav")

# Recognize Text from Audio
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
model.config.forced_decoder_ids = None

# load the audio file to clone into different language
sample, sample_rate = librosa.load(input_audio, sr=16000)

# Split the audio into 25-second segments
segment_length = 25  # in seconds
segment_samples = int(segment_length * sample_rate)

# Initialize a list to store transcriptions for each segment
transcriptions = []

for i in range(0, len(sample), segment_samples):
    segment = sample[i:i+segment_samples]
    input_features = processor(segment, sampling_rate=sample_rate, return_tensors="pt").input_features 
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    transcriptions.append(transcription)
    
# Concatenate transcriptions for all segments
full_transcription = ""
for x in transcriptions:
    full_transcription += x[0]


# Translate English text into Hindi
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": """### Translation Expert Prompt ###

As a proficient language translation model, efficiently translate the following English text into Hindi while maintaining contextual accuracy and linguistic nuance.

---

**Prompt:**
Translate the provided English text into Hindi.

---

**Instructions:**

1. Ensure the translation is grammatically correct and culturally appropriate for a Hindi-speaking audience.

2. Pay attention to preserving the tone and nuances of the original text.

3. Aim for clarity and coherence in the translated output.

4. Provide the translation in a format that is easy to understand and ready for immediate use.

---

By following these instructions, you'll contribute to the creation of high-quality translations that serve the needs of the target audience effectively."""},
    {"role": "user", "content": full_transcription}
  ]
)
translated_text = completion.choices[0].message.content


# Split the translated text into meaningful chunks (e.g., sentences)
chunk_size = 250
translated_chunks = [translated_text[i:i + chunk_size] for i in range(0, len(translated_text), chunk_size)]

# Initialize a list to store audio files for each chunk
audio_files = []

# Generate audio for each chunk
for i, chunk in enumerate(translated_chunks):
    output_file_path = f"/home/saravana/VoiceClone/Git_Project/Chunks/chunk_{i + 1}_output.wav"
    tts.tts_to_file(text=chunk, file_path=output_file_path, speaker_wav= train_audio, language="hi")
    audio_files.append(output_file_path)
    
# Concatenate audio files for all chunks
concatenate_audio(audio_files, output_audio_path)
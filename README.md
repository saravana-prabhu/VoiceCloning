# VoiceCloning
<div>
<a href="https://platform.openai.com/docs/introduction"><img alt="Static Badge" src="https://img.shields.io/badge/OpenAI%20-%20%236A7A61%20"></a>
<a href="https://huggingface.co/openai/whisper-large-v2"><img alt="Static Badge" src="https://img.shields.io/badge/WhisperLargev2%20-%20%237A6461%20"></a>
<a href="https://huggingface.co/coqui/XTTS-v2"><img alt="Static Badge" src="https://img.shields.io/badge/XTTSv2%20-%20%2370617A"></a>
</div>


![AI](https://github.com/saravanasevenn/VoiceCloning/assets/100367006/bde577ee-9d23-47f4-97ca-b434f0f777da)


## Overview

This project aims to create a voice cloning model capable of generating a synthetic voice that mimics the unique vocal characteristics of a specific person. The focus is on cloning a speaker's voice from an English audio source to Hindi while preserving the pitch, tone, and other vocal nuances.

## Purpose

The purpose of this project is to demonstrate the capability of voice cloning for multilingual applications, particularly from English to Hindi. The voice cloning process involves Automatic Speech Recognition (ASR), machine translation using OpenAI's GPT, and training a Text-to-Speech (TTS) model to reproduce the cloned voice.

## Key Features

* Automatic Speech Recognition (ASR) using Whisper Large V2 for English text extraction.
* Translation of English text to Hindi using OpenAI's GPT-3.5-turbo.
* Training a multilingual Text-to-Speech (TTS) model (XTTS) to generate the cloned voice.
* Chunking and concatenation of audio files to overcome model limitations.
* Seamless integration of multiple AI models for an end-to-end voice cloning pipeline.


## Flow of the Project

![voice drawio](https://github.com/saravanasevenn/VoiceCloning/assets/100367006/fef8ba73-5c3a-47e7-85b9-ce39ec058b36)

1. ASR for Text Extraction: Utilize Whisper Large V2 for ASR to extract text from English audio. Split the audio into 25-second segments to overcome model limitations.
2. Translation to Hindi: Use OpenAI's GPT-3.5-turbo to translate the English text into Hindi while maintaining contextual accuracy and linguistic nuance.
3. TTS Model Training: Train the XTTS model in Hindi audio to generate the cloned voice.
4. Voice Cloning Process: Feed the translated text to XTTS in chunks, generating audio for each chunk. Concatenate the audio chunks to obtain the final voice-cloned output in Hindi.

## How to Run the Project

Follow these steps to set up and run the project:

1. Create a virtual environment.
2. Install project dependencies using pip install -r requirements.txt.
3. Obtain your OpenAI API key and enter it in the designated field.
4. Specify file paths for input, training, output, and storage of generated chunks.
5. Run the project to initiate the voice cloning process.


### Installation

clone the repo
```bash
git clone {repo_url}
```

#### Create Environment

```bash
conda create -f environment.yml
```
or use pip
```bash
python -m venv env_name
```
#### Activate Environment
for conda
```bash
conda activate env_name
```
or use pip
```bash
source env_name/bin/activate
```

#### Install Requirements
```bash
pip install -r requirements.txt
```

### Usage and Evaluation

Use your OpenAI api key in app.py

```bash
client = OpenAI(api_key="YOUR_API_KEY")
```
Run the application

```bash
python3 app.py
```


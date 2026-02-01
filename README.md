# Turkish to English Voice Translator

A local, open-source AI voice translation tool that converts Turkish speech to English while preserving the speaker's vocal identity through voice cloning.

## Overview

This application uses a **Decoupled Pipeline** approach to solve a common problem in voice translation: traditional voice conversion keeps the rhythm and prosody of the source language, making translations sound unnatural.

### Pipeline Architecture

```
Turkish Audio --> Transcription --> Translation --> Voice Cloning TTS --> English Audio
                  (faster_whisper)   (Google)        (XTTS v2)
```

1. **Transcription**: Convert Turkish audio to Turkish text using faster_whisper
2. **Translation**: Convert Turkish text to English text using Google Translate
3. **Voice Cloning TTS**: Generate English audio from translated text using the original speaker's voice as a reference

### Key Benefits

- The translated speech has native English prosody (rhythm, intonation)
- The voice timbre is preserved from the original Turkish speaker
- Natural-sounding output without source language rhythm artifacts

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Transcription | faster_whisper |
| Translation | deep_translator (Google Translate) |
| Voice Cloning | Coqui TTS (XTTS v2 model) |

## Requirements

- Python 3.10 or higher
- GPU recommended (CUDA) but CPU is supported
- Approximately 4GB RAM minimum
- 2GB disk space for models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MewIsEpic/SpeechTranslater.git
cd SpeechTranslater
```

2. Create and activate a virtual environment:
```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser to `http://localhost:8501`

3. Upload a Turkish audio file (WAV or MP3)

4. View the transcription and click "Translate and Speak in English"

5. Download the translated audio with your cloned voice

## First Run

On the first run, the application will download the following models:
- Whisper model (~150MB)
- XTTS v2 model (~1.5GB)

This only happens once. Subsequent runs will use the cached models.

## Supported Audio Formats

- WAV
- MP3

## Limitations

- Input audio should be clear with minimal background noise for best results
- Longer audio files will take more time to process
- Voice cloning quality depends on the quality of the input audio
- CPU mode is significantly slower than GPU mode

## Project Structure

```
SpeechTranslater/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── packages.txt        # System dependencies (for cloud deployment)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## License

This project is open source and available for personal and educational use.

## Acknowledgments

- [Coqui TTS](https://github.com/coqui-ai/TTS) for the XTTS v2 voice cloning model
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for efficient speech recognition
- [deep-translator](https://github.com/nidhaloff/deep-translator) for translation services
- [Streamlit](https://streamlit.io/) for the web interface


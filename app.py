"""
Open Source AI Voice Translator
================================
A local voice translation tool using a Decoupled Pipeline approach:

DECOUPLED PIPELINE LOGIC:
-------------------------
Traditional voice conversion keeps the rhythm/prosody of the source language,
making translations sound unnatural. This app solves that by decoupling the process:

1. TRANSCRIPTION: Convert source audio → text (faster_whisper)
2. TRANSLATION: Convert source text → target language text (deep_translator)
3. VOICE CLONING TTS: Generate new audio from translated text using the
   original speaker's voice as a reference (XTTS v2)

This approach ensures:
- The translated speech has NATIVE PROSODY of the target language
- The voice TIMBRE is preserved from the original speaker
- Natural-sounding output that doesn't carry source language rhythm artifacts
"""

import os
import tempfile
import warnings
from pathlib import Path

import streamlit as st
import torch
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel
from TTS.api import TTS

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

# XTTS v2 supported languages with their codes
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Polish": "pl",
    "Turkish": "tr",
    "Russian": "ru",
    "Dutch": "nl",
    "Czech": "cs",
    "Arabic": "ar",
    "Chinese": "zh-cn",
    "Japanese": "ja",
    "Hungarian": "hu",
    "Korean": "ko",
}

# Language code mapping for deep_translator (some differ from XTTS codes)
TRANSLATOR_LANG_CODES = {
    "en": "en",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "pt": "pt",
    "pl": "pl",
    "tr": "tr",
    "ru": "ru",
    "nl": "nl",
    "cs": "cs",
    "ar": "ar",
    "zh-cn": "zh-CN",
    "ja": "ja",
    "hu": "hu",
    "ko": "ko",
}

# =============================================================================
# DEVICE DETECTION
# =============================================================================

def get_device():
    """Check for GPU availability and return appropriate device."""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

DEVICE = get_device()

# =============================================================================
# MODEL LOADING (Cached to load only once)
# =============================================================================

@st.cache_resource
def load_whisper_model():
    """
    Load the faster_whisper model for transcription.
    Using 'base' model for balance between speed and accuracy on limited compute.
    Cached with @st.cache_resource to prevent reloading on each interaction.
    """
    # Use smaller model for CPU, larger for GPU
    model_size = "base" if DEVICE == "cpu" else "small"
    compute_type = "int8" if DEVICE == "cpu" else "float16"
    
    model = WhisperModel(
        model_size,
        device=DEVICE,
        compute_type=compute_type
    )
    return model


@st.cache_resource
def load_tts_model():
    """
    Load the XTTS v2 model for voice cloning and text-to-speech.
    This model can clone voices from a reference audio while generating
    speech with native prosody in the target language.
    Cached with @st.cache_resource to prevent reloading on each interaction.
    """
    tts = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=False
    ).to(DEVICE)
    return tts


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_uploaded_file(uploaded_file) -> str:
    """
    Save the uploaded Streamlit file to a temporary location.
    XTTS requires a file path, not a bytes object, so we must save to disk.
    
    Returns:
        str: Path to the saved temporary file
    """
    # Create temp file with correct extension
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name


def transcribe_audio(audio_path: str, whisper_model) -> tuple[str, str]:
    """
    Transcribe audio file using faster_whisper.
    
    Args:
        audio_path: Path to the audio file
        whisper_model: Loaded WhisperModel instance
        
    Returns:
        tuple: (detected_language, transcribed_text)
    """
    segments, info = whisper_model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True  # Filter out silence for efficiency
    )
    
    # Combine all segments into full transcription
    transcription = " ".join([segment.text for segment in segments])
    
    return info.language, transcription.strip()


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text using Google Translate via deep_translator.
    
    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code (XTTS format)
        
    Returns:
        str: Translated text
    """
    # Convert XTTS lang code to translator code if needed
    target_code = TRANSLATOR_LANG_CODES.get(target_lang, target_lang)
    
    translator = GoogleTranslator(source=source_lang, target=target_code)
    translated = translator.translate(text)
    
    return translated


def generate_cloned_speech(text: str, reference_audio: str, target_lang: str, tts_model) -> str:
    """
    Generate speech in target language while cloning the voice from reference audio.
    
    This is the key step in the decoupled pipeline:
    - The TEXT provides the content and determines the PROSODY (rhythm, intonation)
    - The REFERENCE AUDIO provides the VOICE TIMBRE (tone, characteristics)
    
    Args:
        text: Translated text to speak
        reference_audio: Path to original audio for voice cloning
        target_lang: Target language code
        tts_model: Loaded TTS model instance
        
    Returns:
        str: Path to generated audio file
    """
    # Create output temp file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    
    # Generate speech with voice cloning
    # speaker_wav: reference audio for voice characteristics
    # language: target language for correct prosody
    tts_model.tts_to_file(
        text=text,
        file_path=output_path,
        speaker_wav=reference_audio,
        language=target_lang
    )
    
    return output_path


# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Voice Translator",
        page_icon="🎙️",
        layout="centered"
    )
    
    # Custom styling
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
        }
        .step-header {
            color: #4A90A4;
            margin-top: 1.5rem;
        }
        .info-box {
            background-color: #1E1E1E;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #4A90A4;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown("<h1 class='main-header'>🎙️ Open Source AI Voice Translator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Translate voice recordings while preserving your vocal identity</p>", unsafe_allow_html=True)
    
    # Device status indicator
    if DEVICE == "cuda":
        st.success("✅ GPU (CUDA) detected - Using hardware acceleration")
    else:
        st.warning("⚠️ No GPU detected - Running on CPU (slower performance)")
    
    st.divider()
    
    # ---------------------------------------------------------------------
    # STEP 1: File Upload
    # ---------------------------------------------------------------------
    st.markdown("### 📁 Step 1: Upload Audio")
    uploaded_file = st.file_uploader(
        "Upload a voice recording (WAV or MP3)",
        type=["wav", "mp3"],
        help="Upload an audio file containing speech in any supported language"
    )
    
    if uploaded_file is not None:
        # Save uploaded file to disk (required for XTTS)
        try:
            audio_path = save_uploaded_file(uploaded_file)
            
            # Store path in session state for later use
            st.session_state["audio_path"] = audio_path
            
        except Exception as e:
            st.error(f"❌ Error saving file: {str(e)}")
            return
        
        # -----------------------------------------------------------------
        # STEP 2: Display Original Audio
        # -----------------------------------------------------------------
        st.markdown("### 🔊 Step 2: Original Audio")
        st.audio(uploaded_file, format=f"audio/{Path(uploaded_file.name).suffix[1:]}")
        
        # -----------------------------------------------------------------
        # STEP 3: Transcription
        # -----------------------------------------------------------------
        st.markdown("### 📝 Step 3: Transcription")
        
        with st.spinner("Loading transcription model..."):
            whisper_model = load_whisper_model()
        
        with st.spinner("Transcribing audio..."):
            try:
                detected_lang, transcription = transcribe_audio(audio_path, whisper_model)
                
                # Store in session state
                st.session_state["detected_lang"] = detected_lang
                st.session_state["transcription"] = transcription
                
                # Display results
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Detected Language", detected_lang.upper())
                with col2:
                    st.info(f"**Transcription:** {transcription}")
                    
            except Exception as e:
                st.error(f"❌ Transcription error: {str(e)}")
                return
        
        # -----------------------------------------------------------------
        # STEP 4: Target Language Selection
        # -----------------------------------------------------------------
        st.markdown("### 🌍 Step 4: Select Target Language")
        
        target_language = st.selectbox(
            "Choose the language for translation",
            options=list(SUPPORTED_LANGUAGES.keys()),
            index=0,
            help="Select the language you want the audio translated to"
        )
        
        target_lang_code = SUPPORTED_LANGUAGES[target_language]
        
        # -----------------------------------------------------------------
        # STEP 5: Translate & Generate
        # -----------------------------------------------------------------
        st.markdown("### 🚀 Step 5: Generate Translation")
        
        translate_button = st.button(
            "🎯 Translate & Speak",
            type="primary",
            use_container_width=True
        )
        
        if translate_button:
            # Check if we have required data
            if "transcription" not in st.session_state:
                st.error("❌ Please wait for transcription to complete")
                return
                
            transcription = st.session_state["transcription"]
            detected_lang = st.session_state["detected_lang"]
            audio_path = st.session_state["audio_path"]
            
            # Check if source and target are the same
            if detected_lang == target_lang_code:
                st.warning("⚠️ Source and target language are the same. Select a different target language.")
                return
            
            # Translation step
            with st.spinner("Translating text..."):
                try:
                    translated_text = translate_text(
                        transcription,
                        detected_lang,
                        target_lang_code
                    )
                    st.success(f"**Translated Text ({target_language}):** {translated_text}")
                    
                except Exception as e:
                    st.error(f"❌ Translation error: {str(e)}")
                    return
            
            # Voice cloning TTS step
            with st.spinner("Loading voice cloning model (this may take a moment on first run)..."):
                tts_model = load_tts_model()
            
            with st.spinner("Generating cloned voice audio..."):
                try:
                    # This is the DECOUPLED PIPELINE in action:
                    # - translated_text: provides content + native prosody
                    # - audio_path: provides voice timbre for cloning
                    output_path = generate_cloned_speech(
                        translated_text,
                        audio_path,
                        target_lang_code,
                        tts_model
                    )
                    
                    # Display the result
                    st.markdown("---")
                    st.markdown("### 🎧 Translated Audio Output")
                    st.audio(output_path, format="audio/wav")
                    
                    # Download button
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Translated Audio",
                            data=f,
                            file_name=f"translated_{target_language.lower()}.wav",
                            mime="audio/wav"
                        )
                    
                    st.success("✅ Translation complete!")
                    
                    # Cleanup temp output file reference (file still accessible for download)
                    st.session_state["output_path"] = output_path
                    
                except Exception as e:
                    st.error(f"❌ Voice generation error: {str(e)}")
                    return
    
    # Footer
    st.divider()
    st.markdown("""
        <p style='text-align: center; color: #666; font-size: 0.8rem;'>
        Powered by faster_whisper, deep_translator, and Coqui TTS (XTTS v2)<br>
        All processing happens locally on your machine
        </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


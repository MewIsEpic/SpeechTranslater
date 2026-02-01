"""
Turkish to English Voice Translator
====================================
A local voice translation tool using a Decoupled Pipeline approach:

DECOUPLED PIPELINE LOGIC:
-------------------------
Traditional voice conversion keeps the rhythm/prosody of the source language,
making translations sound unnatural. This app solves that by decoupling the process:

1. TRANSCRIPTION: Convert Turkish audio → Turkish text (faster_whisper)
2. TRANSLATION: Convert Turkish text → English text (deep_translator)
3. VOICE CLONING TTS: Generate English audio from translated text using the
   original speaker's voice as a reference (XTTS v2)

This approach ensures:
- The translated speech has NATIVE ENGLISH PROSODY
- The voice TIMBRE is preserved from the original Turkish speaker
- Natural-sounding output that doesn't carry Turkish rhythm artifacts
"""

import os
import tempfile
import warnings
from pathlib import Path

# Auto-accept Coqui TTS Terms of Service (required for non-interactive environments)
os.environ["COQUI_TOS_AGREED"] = "1"

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

# Fixed language settings for Turkish → English translation
SOURCE_LANG = "tr"  # Turkish
TARGET_LANG = "en"  # English

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
    Transcribe Turkish audio file using faster_whisper.
    
    Args:
        audio_path: Path to the audio file
        whisper_model: Loaded WhisperModel instance
        
    Returns:
        tuple: (detected_language, transcribed_text)
    """
    segments, info = whisper_model.transcribe(
        audio_path,
        beam_size=5,
        language="tr",  # Force Turkish detection for accuracy
        vad_filter=True  # Filter out silence for efficiency
    )
    
    # Combine all segments into full transcription
    transcription = " ".join([segment.text for segment in segments])
    
    return info.language, transcription.strip()


def translate_text(text: str) -> str:
    """
    Translate Turkish text to English using Google Translate via deep_translator.
    
    Args:
        text: Turkish text to translate
        
    Returns:
        str: English translated text
    """
    translator = GoogleTranslator(source="tr", target="en")
    translated = translator.translate(text)
    
    return translated


def generate_cloned_speech(text: str, reference_audio: str, tts_model) -> str:
    """
    Generate English speech while cloning the voice from Turkish reference audio.
    
    This is the key step in the decoupled pipeline:
    - The TEXT provides the content and determines the PROSODY (rhythm, intonation)
    - The REFERENCE AUDIO provides the VOICE TIMBRE (tone, characteristics)
    
    Args:
        text: English translated text to speak
        reference_audio: Path to original Turkish audio for voice cloning
        tts_model: Loaded TTS model instance
        
    Returns:
        str: Path to generated audio file
    """
    # Create output temp file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    
    # Generate speech with voice cloning
    # speaker_wav: reference audio for voice characteristics
    # language: English for correct prosody
    tts_model.tts_to_file(
        text=text,
        file_path=output_path,
        speaker_wav=reference_audio,
        language="en"  # Generate with English prosody
    )
    
    return output_path


# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Turkish → English Voice Translator",
        page_icon="🇹🇷",
        layout="centered"
    )
    
    # Custom styling
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
        }
        .lang-badge {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 1rem;
            font-weight: bold;
            margin: 0.2rem;
        }
        .tr-badge {
            background: linear-gradient(135deg, #E30A17 0%, #ff4444 100%);
            color: white;
        }
        .en-badge {
            background: linear-gradient(135deg, #012169 0%, #4466aa 100%);
            color: white;
        }
        .arrow {
            font-size: 1.5rem;
            margin: 0 0.5rem;
        }
        .step-header {
            color: #E30A17;
            margin-top: 1.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title with language badges
    st.markdown("<h1 class='main-header'>🎙️ Turkish → English Voice Translator</h1>", unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align: center;'>
            <span class='lang-badge tr-badge'>🇹🇷 Türkçe</span>
            <span class='arrow'>→</span>
            <span class='lang-badge en-badge'>🇬🇧 English</span>
        </p>
        <p style='text-align: center; color: #888;'>Translate Turkish voice recordings to English while preserving your vocal identity</p>
    """, unsafe_allow_html=True)
    
    # Device status indicator
    if DEVICE == "cuda":
        st.success("✅ GPU (CUDA) detected - Using hardware acceleration")
    else:
        st.warning("⚠️ No GPU detected - Running on CPU (slower performance)")
    
    st.divider()
    
    # ---------------------------------------------------------------------
    # STEP 1: File Upload
    # ---------------------------------------------------------------------
    st.markdown("### 📁 Step 1: Upload Turkish Audio")
    uploaded_file = st.file_uploader(
        "Upload a Turkish voice recording (WAV or MP3)",
        type=["wav", "mp3"],
        help="Upload an audio file containing Turkish speech"
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
        st.markdown("### 🔊 Step 2: Original Turkish Audio")
        st.audio(uploaded_file, format=f"audio/{Path(uploaded_file.name).suffix[1:]}")
        
        # -----------------------------------------------------------------
        # STEP 3: Transcription
        # -----------------------------------------------------------------
        st.markdown("### 📝 Step 3: Turkish Transcription")
        
        with st.spinner("Loading transcription model..."):
            whisper_model = load_whisper_model()
        
        with st.spinner("Transcribing Turkish audio..."):
            try:
                detected_lang, transcription = transcribe_audio(audio_path, whisper_model)
                
                # Store in session state
                st.session_state["transcription"] = transcription
                
                # Display results
                st.info(f"**🇹🇷 Turkish Transcription:** {transcription}")
                
                # Warn if not Turkish
                if detected_lang != "tr":
                    st.warning(f"⚠️ Detected language: {detected_lang.upper()}. This app is optimized for Turkish input.")
                    
            except Exception as e:
                st.error(f"❌ Transcription error: {str(e)}")
                return
        
        # -----------------------------------------------------------------
        # STEP 4: Translate & Generate
        # -----------------------------------------------------------------
        st.markdown("### 🚀 Step 4: Translate to English")
        
        translate_button = st.button(
            "🎯 Translate & Speak in English",
            type="primary",
            use_container_width=True
        )
        
        if translate_button:
            # Check if we have required data
            if "transcription" not in st.session_state:
                st.error("❌ Please wait for transcription to complete")
                return
                
            transcription = st.session_state["transcription"]
            audio_path = st.session_state["audio_path"]
            
            # Translation step
            with st.spinner("Translating to English..."):
                try:
                    translated_text = translate_text(transcription)
                    st.success(f"**🇬🇧 English Translation:** {translated_text}")
                    
                except Exception as e:
                    st.error(f"❌ Translation error: {str(e)}")
                    return
            
            # Voice cloning TTS step
            with st.spinner("Loading voice cloning model (this may take a moment on first run)..."):
                tts_model = load_tts_model()
            
            with st.spinner("Generating English audio with your voice..."):
                try:
                    # This is the DECOUPLED PIPELINE in action:
                    # - translated_text: provides English content + native English prosody
                    # - audio_path: provides Turkish speaker's voice timbre for cloning
                    output_path = generate_cloned_speech(
                        translated_text,
                        audio_path,
                        tts_model
                    )
                    
                    # Display the result
                    st.markdown("---")
                    st.markdown("### 🎧 English Audio Output (Your Voice)")
                    st.audio(output_path, format="audio/wav")
                    
                    # Download button
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download English Audio",
                            data=f,
                            file_name="turkish_to_english_translation.wav",
                            mime="audio/wav"
                        )
                    
                    st.success("✅ Translation complete! Your Turkish speech is now in English with your voice.")
                    
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
        🇹🇷 Turkish → 🇬🇧 English | All processing happens locally
        </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

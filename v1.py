import os
import io
import json
import tempfile
import traceback
from typing import List, Optional

import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pydub import AudioSegment
from langdetect import detect, DetectorFactory
import speech_recognition as sr
from audiorecorder import audiorecorder   # from streamlit-audiorecorder

# ---------------- CONFIG ----------------
DetectorFactory.seed = 0
MODEL_NAME = "google/flan-t5-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_T5_INPUT = 512

@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -------------- HELPER FUNCTIONS -----------------
def run_t5_prompt(prompt: str, max_length: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_T5_INPUT)
    input_ids = inputs.input_ids.to(DEVICE)
    attention_mask = inputs.attention_mask.to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=3,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def convert_to_wav_mono_16k(input_bytes: bytes, input_filename: str = "input.wav") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(input_filename)[1]) as in_f:
        in_f.write(input_bytes)
        in_path = in_f.name
    try:
        audio = AudioSegment.from_file(in_path)
    except Exception:
        audio = AudioSegment.from_wav(in_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)
    audio.export(out_path, format="wav")
    os.remove(in_path)
    return out_path

def transcribe_with_google(file_path: str, language_code: str = "en-US"):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data, language=language_code)
        return text, {"engine": "google_web_speech", "language": language_code}
    except Exception as e:
        return "", {"engine": "google_web_speech", "error": str(e)}

def detect_language_from_text(text: str) -> Optional[str]:
    if not text.strip():
        return None
    try:
        code = detect(text)
        if code.startswith("en"): return "en"
        if code.startswith("hi"): return "hi"
        if code.startswith("mr"): return "mr"
        return code
    except Exception:
        return None

# --- Prompt Templates ---
def prompt_extract_qa(t: str): 
    return f"""Extract Q&A pairs (doctor/patient) from the conversation:
{t}
Return JSON array of objects with fields: speaker_question, question, answer, timestamps."""

def prompt_extract_entities(t: str):
    return f"""Extract medical entities (symptoms, diagnosis, medication, etc.) from:
{t}
Return JSON array with fields: term, type, suggested_code, confidence."""

def prompt_summary(t: str):
    return f"""Summarize this doctor-patient conversation (3 sentences max):
{t}"""

def extract_qa(transcript: str) -> List[dict]:
    result = run_t5_prompt(prompt_extract_qa(transcript), max_length=384)
    try:
        return json.loads(result)
    except Exception:
        return [{"speaker_question": "doctor/patient", "question": "", "answer": transcript, "timestamps": ""}]

def extract_entities(text: str) -> List[dict]:
    result = run_t5_prompt(prompt_extract_entities(text), max_length=256)
    try:
        return json.loads(result)
    except Exception:
        return []

def generate_summary(text: str) -> str:
    return run_t5_prompt(prompt_summary(text), max_length=128).strip()

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="Med-Conversation POC", page_icon="🩺", layout="centered")
st.title("🩺 Med-Conversation POC")
st.caption("Upload or record doctor-patient audio → Transcription → Language detection → Q&A → Entities → Summary")

# Tabs for input methods
tab1, tab2 = st.tabs(["🎙️ Record audio", "📁 Upload audio"])

audio_bytes = None
input_name = "input.wav"

with tab1:
    st.markdown("Press record, then stop when done.")
    recorded_audio = audiorecorder("Start Recording", "Stop Recording")
'''
    if len(recorded_audio) > 0:
        st.audio(recorded_audio.tobytes(), format="audio/wav")
        audio_bytes = recorded_audio.tobytes()
        input_name = "mic_recording.wav"
'''


    if len(recorded_audio) > 0:
    # Convert AudioSegment → WAV bytes
        buf = io.BytesIO()
        recorded_audio.export(buf, format="wav")
        audio_bytes = buf.getvalue()
        st.audio(audio_bytes, format="audio/wav")
        input_name = "mic_recording.wav"




with tab2:
    uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a"])
    if uploaded_file:
        st.audio(uploaded_file)
        audio_bytes = uploaded_file.read()
        input_name = uploaded_file.name

prefer_language = st.selectbox("Preferred language", ["auto-detect", "en", "hi", "mr"])

if audio_bytes:
    if st.button("🚀 Process"):
        with st.spinner("Processing audio... This may take a few seconds ⏳"):
            try:
                wav_path = convert_to_wav_mono_16k(audio_bytes, input_name)
                first_text, log1 = transcribe_with_google(wav_path, "en-US")
                detected = detect_language_from_text(first_text)
                if prefer_language != "auto-detect":
                    detected = prefer_language

                lang_map = {"en": "en-US", "hi": "hi-IN", "mr": "mr-IN"}
                lang_code = lang_map.get(detected, "en-US")

                if detected in ("hi", "mr"):
                    final_text, log2 = transcribe_with_google(wav_path, lang_code)
                else:
                    final_text, log2 = first_text, {}

                if not final_text.strip():
                    st.error("No transcription obtained.")
                    st.stop()

                st.subheader("📝 Transcript")
                st.write(final_text)

                st.subheader("🌐 Detected language")
                st.write(f"Detected: **{detected or 'unknown'}** | Google code: `{lang_code}`")

                qa_pairs = extract_qa(final_text)
                entities = []
                for qa in qa_pairs:
                    combo = f"{qa.get('question','')} {qa.get('answer','')}"
                    ents = extract_entities(combo)
                    qa["entities"] = ents
                    entities.extend(ents)

                summary = generate_summary(final_text)

                st.subheader("❓ Extracted Q&A")
                st.json(qa_pairs)

                st.subheader("🏷️ Entities")
                st.json(entities)

                st.subheader("📋 Summary")
                st.write(summary)

                os.remove(wav_path)

            except Exception as e:
                st.error(f"Error: {e}")
                st.text(traceback.format_exc())


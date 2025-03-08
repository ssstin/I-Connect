import streamlit as st
import pandas as pd
import numpy as np
import time
import cv2 as cv
import mediapipe as mp
import copy
import itertools
from collections import Counter, deque

from speechtospeech import Can_to_eng, Eng_to_can, MultilingualTranslator
from cv.utils.cvfpscalc import CvFpsCalc
from cv.model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from cv.model.point_history_classifier.point_history_classifier import PointHistoryClassifier

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Speech Translator",
        page_icon="🗣️",
        layout="centered"
    )
    
    # App title with styling
    st.markdown("""
    # 🗣️ Speech Translator
    Translate between Cantonese-English or any supported languages using your voice
    """)
    
    # Create session state variables
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
    if 'output_text' not in st.session_state:
        st.session_state.output_text = ""
    if 'translation_count' not in st.session_state:
        st.session_state.translation_count = 0
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Translator", "Hand Gesture", "History"])
    
    with tab1:
        st.subheader("Select Translation Mode")
        mode = st.radio(
            "Choose a mode:",
            ("Cantonese-English Translation", "International Translation")
        )
        
        if mode == "Cantonese-English Translation":
            st.subheader("Select Translation Direction")
            choice = st.radio(
                "Choose an option:",
                ("Speak in Cantonese → Translate to English",
                 "Speak in English → Translate to Cantonese")
            )
            
            # Translation section
            col1, col2 = st.columns([2, 1])
            with col1:
                translate_button = st.button("Start Recording & Translate", type="primary", key="can_eng")
            with col2:
                clear_button = st.button("Clear Results", key="clear_can_eng")
                if clear_button:
                    st.session_state.input_text = ""
                    st.session_state.output_text = ""
            
            if translate_button:
                with st.status("Processing...", expanded=True) as status:
                    st.write("🎤 Recording audio...")
                    
                    if choice == "Speak in Cantonese → Translate to English":
                        cantonese_text, english_translation = Can_to_eng.translate_cantonese_to_english_speech_with_return()
                        st.session_state.input_text = cantonese_text if cantonese_text else "No speech detected"
                        st.session_state.output_text = english_translation if english_translation else "Translation failed"
                    else:
                        english_text, cantonese_translation = Eng_to_can.translate_english_to_cantonese_speech_with_return()
                        st.session_state.input_text = english_text if english_text else "No speech detected"
                        st.session_state.output_text = cantonese_translation if cantonese_translation else "Translation failed"
                    
                    st.write("✅ Translation complete!")
                    status.update(label="Translation complete!", state="complete")
                    
                    # Add to history
                    if st.session_state.input_text and st.session_state.output_text:
                        timestamp = time.strftime("%H:%M:%S")
                        st.session_state.history.append({
                            "timestamp": timestamp,
                            "direction": choice,
                            "input": st.session_state.input_text,
                            "output": st.session_state.output_text
                        })
                        st.session_state.translation_count += 1
            
            # Display results
            st.subheader("Translation Results")
            results_cols = st.columns(2)
            with results_cols[0]:
                source_lang = "Cantonese" if choice == "Speak in Cantonese → Translate to English" else "English"
                st.markdown(f"##### Source: {source_lang}")
                st.info(st.session_state.input_text if st.session_state.input_text else "Input will appear here")
            with results_cols[1]:
                target_lang = "English" if choice == "Speak in Cantonese → Translate to English" else "Cantonese"
                st.markdown(f"##### Target: {target_lang}")
                st.success(st.session_state.output_text if st.session_state.output_text else "Translation will appear here")
        
        elif mode == "International Translation":
            st.subheader("Select Languages")
            # Expanded language list including Chinese (Mandarin and Cantonese)
            languages = {
                "Arabic": "ar",
                "Bahasa (Indonesian)": "id",
                "Bengali": "bn",
                "Bulgarian": "bg",
                "Chinese (Mandarin)": "zh",
                "Croatian": "hr",
                "Czech": "cs",
                "Danish": "da",
                "Dutch": "nl",
                "English": "en",
                "Finnish": "fi",
                "French": "fr",
                "German": "de",
                "Greek": "el",
                "Hindi": "hi",
                "Hungarian": "hu",
                "Indonesian": "id",
                "Italian": "it",
                "Japanese": "ja",
                "Korean": "ko",
                "Malay": "ms",
                "Norwegian": "no",
                "Polish": "pl",
                "Portuguese": "pt",
                "Romanian": "ro",
                "Russian": "ru",
                "Spanish": "es",
                "Swahili": "sw",
                "Swedish": "sv",
                "Tamil": "ta",
                "Thai": "th",
                "Turkish": "tr",
                "Vietnamese": "vi"
            }
            
            col1, col2 = st.columns(2)
            with col1:
                source_lang = st.selectbox("Source Language", list(languages.keys()), key="source_intl")
            with col2:
                target_lang = st.selectbox("Target Language", list(languages.keys()), index=9, key="target_intl")  # Default to English
            
            # Translation section
            col1, col2 = st.columns([2, 1])
            with col1:
                translate_button = st.button("Start Recording & Translate", type="primary", key="intl")
            with col2:
                clear_button = st.button("Clear Results", key="clear_intl")
                if clear_button:
                    st.session_state.input_text = ""
                    st.session_state.output_text = ""
            
            if translate_button:
                with st.status("Processing...", expanded=True) as status:
                    st.write("🎤 Recording audio...")
                    
                    translator = MultilingualTranslator()
                    source_code = languages[source_lang]
                    target_code = languages[target_lang]
                    
                    input_text, translated_text = translator.translate_speech(
                        source_lang=source_code, 
                        target_lang=target_code
                    )
                    
                    st.session_state.input_text = input_text if input_text else "No speech detected"
                    st.session_state.output_text = translated_text if translated_text else "Translation failed"
                    
                    st.write("✅ Translation complete!")
                    status.update(label="Translation complete!", state="complete")
                    
                    # Add to history
                    if st.session_state.input_text and st.session_state.output_text:
                        timestamp = time.strftime("%H:%M:%S")
                        st.session_state.history.append({
                            "timestamp": timestamp,
                            "direction": f"{source_lang} → {target_lang}",
                            "input": st.session_state.input_text,
                            "output": st.session_state.output_text
                        })
                        st.session_state.translation_count += 1
            
            # Display results
            st.subheader("Translation Results")
            results_cols = st.columns(2)
            with results_cols[0]:
                st.markdown(f"##### Source: {source_lang}")
                st.info(st.session_state.input_text if st.session_state.input_text else "Input will appear here")
            with results_cols[1]:
                st.markdown(f"##### Target: {target_lang}")
                st.success(st.session_state.output_text if st.session_state.output_text else "Translation will appear here")
    
    # Hand Gesture tab (unchanged for brevity)
    with tab2:
        st.subheader("Hand Gesture Recognition")
        # Your existing hand gesture code here (omitted for brevity)
        st.write("Use hand gestures to control the application (unchanged)")
    
    # History tab
    with tab3:
        st.subheader(f"Translation History ({st.session_state.translation_count})")
        if not st.session_state.history:
            st.info("No translations yet. Start translating to build your history!")
        else:
            for i, item in enumerate(reversed(st.session_state.history)):
                with st.expander(f"{item['timestamp']} - {item['direction']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Input:**")
                        st.info(item['input'])
                    with col2:
                        st.markdown("**Output:**")
                        st.success(item['output'])

# Helper functions (unchanged, omitted for brevity)
def calc_bounding_rect(image, landmarks): pass  # Your existing function
def calc_landmark_list(image, landmarks): pass  # Your existing function
def pre_process_landmark(landmark_list): pass  # Your existing function
def pre_process_point_history(image, point_history): pass  # Your existing function
def draw_landmarks(image, landmark_point): pass  # Your existing function
def draw_bounding_rect(use_brect, image, brect): pass  # Your existing function
def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text): pass  # Your existing function
def draw_point_history(image, point_history): pass  # Your existing function
def draw_info(image, fps, mode, number): pass  # Your existing function

if __name__ == "__main__":
    main()
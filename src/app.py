import streamlit as st
import pandas as pd
import numpy as np
import time

from speechtospeech import Can_to_eng, Eng_to_can


def main():
    # Set page configuration
    st.set_page_config(
        page_title="Cantonese-English Translator",
        page_icon="🗣️",
        layout="centered"
    )
    
    # App title with styling
    st.markdown("""
    # 🗣️ Cantonese-English Speech Translator
    Translate between Cantonese and English using your voice
    """)
    
    # Create session state variables if they don't exist
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
    if 'output_text' not in st.session_state:
        st.session_state.output_text = ""
    if 'translation_count' not in st.session_state:
        st.session_state.translation_count = 0
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Translator", "History"])
    
    with tab1:
        # Language selection
        st.subheader("Select Translation Direction")
        choice = st.radio(
            "Choose an option:",
            ("Speak in Cantonese → Translate to English",
             "Speak in English → Translate to Cantonese")
        )
        
        # Translation section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            translate_button = st.button("Start Recording & Translate", type="primary")
        
        with col2:
            clear_button = st.button("Clear Results")
            if clear_button:
                st.session_state.input_text = ""
                st.session_state.output_text = ""
        
        # Status indicator and processing feedback
        if translate_button:
            with st.status("Processing...", expanded=True) as status:
                st.write("🎤 Recording audio...")
                
                if choice == "Speak in Cantonese → Translate to English":
                    # Capture the returned values
                    cantonese_text, english_translation = Can_to_eng.translate_cantonese_to_english_speech_with_return()
                    
                    # Update session state
                    st.session_state.input_text = cantonese_text if cantonese_text else "No speech detected"
                    st.session_state.output_text = english_translation if english_translation else "Translation failed"
                    
                elif choice == "Speak in English → Translate to Cantonese":
                    # Capture the returned values
                    english_text, cantonese_translation = Eng_to_can.translate_english_to_cantonese_speech_with_return()
                    
                    # Update session state
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
        
    # Display results in cards
    st.subheader("Translation Results")
    results_cols = st.columns(2)
    
    with results_cols[0]:
        # Correctly identify source language based on translation direction
        source_lang = "Cantonese" if choice == "Speak in Cantonese → Translate to English" else "English"
        st.markdown(f"##### Source: {source_lang}")
        st.info(st.session_state.input_text if st.session_state.input_text else "Input will appear here")
        
    with results_cols[1]:
        # Correctly identify target language based on translation direction
        target_lang = "English" if choice == "Speak in Cantonese → Translate to English" else "Cantonese"
        st.markdown(f"##### Target: {target_lang}")
        st.success(st.session_state.output_text if st.session_state.output_text else "Translation will appear here")
            
    # History tab content
    with tab2:
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

if __name__ == "__main__":
    main()
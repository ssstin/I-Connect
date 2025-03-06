import streamlit as st
import pandas as pd
import numpy as np

from speechtospeech.can_to_eng import translate_cantonese_to_english_speech
from speechtospeech.eng_to_can import translate_english_to_cantonese_speech

def main():
    st.title("Cantonese-English Speech Translator")

    choice = st.radio(
        "Choose an option:",
        ("Speak in Cantonese and translate to English speech",
         "Speak in English and translate to Cantonese speech")
    )

    if st.button("Translate"):
        if choice == "Speak in Cantonese and translate to English speech":
            translate_cantonese_to_english_speech()
        elif choice == "Speak in English and translate to Cantonese speech":
            translate_english_to_cantonese_speech()

if __name__ == "__main__":
    main()

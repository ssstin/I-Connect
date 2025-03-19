import speech_recognition as sr
from gtts import gTTS
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, MarianMTModel, MarianTokenizer
from pydub import AudioSegment
import numpy as np
import os
import tempfile
import pygame

class Can_to_eng:
    recognizer = sr.Recognizer()

    # Load ASR model for Cantonese
    model_name = "CAiRE/wav2vec2-large-xlsr-53-cantonese"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    # Load translation model (Cantonese to English)
    translation_model_name = "Helsinki-NLP/opus-mt-zh-en"
    translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    translation_model = MarianMTModel.from_pretrained(translation_model_name)

    @classmethod
    def translate_audio(cls, audio_path):
        """
        Processes an audio file, transcribes it in Cantonese, translates to English, and returns the text.
        """
        print(f"✅ Translating Cantonese audio: {audio_path}")

        # 1. Load and process audio file
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        samples = np.array(audio.get_array_of_samples())

        # 2. Convert to tensor
        audio_input = torch.tensor(samples, dtype=torch.float32) / 32768.0  # Normalize

        # 3. ASR (Cantonese Speech to Text)
        inputs = cls.processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = cls.model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        cantonese_text = cls.processor.batch_decode(predicted_ids)[0]

        # 4. Translate Cantonese to English
        translated_text = cls.translate_text(cantonese_text)

        print(f"✅ Transcription (Cantonese): {cantonese_text}")
        print(f"✅ Translation (English): {translated_text}")

        # 5. Convert translated text to speech
        cls.speak_text(translated_text)

        return cantonese_text, translated_text

    @classmethod
    def translate_text(cls, cantonese_text):
        """Translates Cantonese text to English"""
        translated = cls.translation_tokenizer(cantonese_text, return_tensors="pt", padding=True)
        generated_tokens = cls.translation_model.generate(**translated)
        translation = cls.translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translation

    @classmethod
    def speak_text(cls, text):
        """Converts text to speech using gTTS"""
        print(f"🔊 Speaking: {text}")
        tts = gTTS(text=text, lang="en")
        temp_filename = tempfile.mktemp(suffix=".mp3")
        tts.save(temp_filename)
        pygame.mixer.init()
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()

import speech_recognition as sr
from gtts import gTTS
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, M2M100ForConditionalGeneration, M2M100Tokenizer
from pydub import AudioSegment
import numpy as np
import os
import tempfile
import pygame

class Eng_to_can:
    recognizer = sr.Recognizer()

    # Load ASR model for English
    model_name = "facebook/wav2vec2-large-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    # Load translation model (English to Chinese)
    translation_model_name = "facebook/m2m100_418M"
    translation_tokenizer = M2M100Tokenizer.from_pretrained(translation_model_name)
    translation_model = M2M100ForConditionalGeneration.from_pretrained(translation_model_name)

    @classmethod
    def translate_audio(cls, audio_path):
        """
        Processes an audio file, transcribes it in English, translates to Chinese, and returns the text.
        """
        print(f"✅ Translating English audio: {audio_path}")

        try:
            # 1. Load and process audio file
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

            # Normalize audio (keeping as float32)
            samples = samples / 32768.0  # Normalize 16-bit audio

            # Do NOT add extra dimensions - keep as 1D array for wav2vec input
            audio_input = torch.tensor(samples)

            # 3. ASR (English Speech to Text)
            inputs = cls.processor(audio_input, sampling_rate=16000, return_tensors="pt")
            
            with torch.no_grad():
                logits = cls.model(inputs.input_values).logits
                
            predicted_ids = torch.argmax(logits, dim=-1)
            english_text = cls.processor.batch_decode(predicted_ids)[0]

            # Only proceed if we got valid text
            if english_text and english_text.strip():
                # 4. Translate English to Chinese
                translated_text = cls.translate_text(english_text)

                print(f"✅ Transcription (English): {english_text}")
                print(f"✅ Translation (Chinese): {translated_text}")

                # 5. Convert translated text to speech
                cls.speak_text(translated_text)

                return english_text, translated_text
            else:
                print("❌ No text was transcribed from the audio")
                return None, None
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    @classmethod
    def translate_text(cls, english_text):
        """Translates English text to Chinese (which includes Cantonese)"""
        cls.translation_tokenizer.src_lang = "en"
        translated = cls.translation_tokenizer(english_text, return_tensors="pt", padding=True)
        
        # Use "zh" for Chinese instead of "yue" for Cantonese
        generated_tokens = cls.translation_model.generate(
            **translated, forced_bos_token_id=cls.translation_tokenizer.get_lang_id("zh")
        )
        translation = cls.translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translation

    @classmethod
    def speak_text(cls, text):
        """Converts text to speech using gTTS"""
        print(f"🔊 Speaking: {text}")
        try:
            # Try Cantonese first (using "yue" code for gTTS)
            tts = gTTS(text=text, lang="yue")
        except ValueError:
            try:
                # Fall back to Chinese if Cantonese not available
                print("⚠️ Cantonese not available, falling back to Mandarin")
                tts = gTTS(text=text, lang="zh-CN")
            except Exception as e:
                print(f"❌ Error with text-to-speech: {str(e)}")
                return
        
        temp_filename = tempfile.mktemp(suffix=".mp3")
        tts.save(temp_filename)
        
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"❌ Error playing audio: {str(e)}")
        finally:
            # Ensure temp file is cleaned up
            try:
                os.remove(temp_filename)
            except:
                pass
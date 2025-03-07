import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from playsound import playsound


class Eng_to_can(object):
    
    @classmethod
    def speakText(cls, command):
        """
        Function to convert text to speech using gTTS.
        Attempts to use Cantonese first, then falls back to Mandarin if needed.

        Args:
            command (str): The text to be spoken.
        """
        print(f"Speaking in Cantonese: {command}")
        
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_filename = temp_file.name
        
        try:
            # First attempt with Cantonese (zh-yue)
            tts = gTTS(text=command, lang='zh-yue')
            tts.save(temp_filename)
            print("Using Cantonese voice")
        except ValueError as e:
            try:
                # Second attempt with alternative Cantonese code
                tts = gTTS(text=command, lang='yue')
                tts.save(temp_filename)
                print("Using Cantonese voice (alternative code)")
            except ValueError as e:
                # Fallback to Mandarin Chinese
                print("Cantonese not available. Falling back to Mandarin Chinese")
                tts = gTTS(text=command, lang='zh-CN')
                tts.save(temp_filename)
        
        # Play the audio file
        playsound(temp_filename)
        
        # Clean up the file after playing
        try:
            os.unlink(temp_filename)
        except:
            pass  # File may be in use, will be cleaned up later

    @classmethod
    def translate_english_to_cantonese_speech(cls):
        """
        Function to recognize speech in English and translate it to Cantonese speech.
        """
        recognizer = sr.Recognizer()

        with sr.Microphone() as source:
            print("Listening in English...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)

            # Listen for the user's input
            audio = recognizer.listen(source)

            try:
                # Using Google Speech Recognition to recognize the audio in English
                recognized_text = recognizer.recognize_google(audio)
                print(f"Recognized Text (English): {recognized_text}")

                # Translate the English text to Cantonese using M2M100
                cantonese_translation = cls.translate_text_with_m2m100(
                    recognized_text, "en", "zh"
                )

                print(f"Translated Text (Cantonese): {cantonese_translation}")

                # Speak out the Cantonese translation
                cls.speakText(cantonese_translation)

                return recognized_text

            except sr.RequestError as e:
                print(f"Could not request results; {e}")
            except sr.UnknownValueError:
                print("Unknown error occurred.")

    @classmethod
    def translate_english_to_cantonese_speech_with_return(cls):
        """
        Function to recognize speech in English, translate it to Cantonese speech,
        and return both the original and translated texts.
        
        Returns:
            tuple: (english_text, cantonese_translation)
        """
        recognizer = sr.Recognizer()

        with sr.Microphone() as source:
            print("Listening in English...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)

            # Listen for the user's input
            audio = recognizer.listen(source)

            try:
                # Using Google Speech Recognition to recognize the audio in English
                recognized_text = recognizer.recognize_google(audio)
                print(f"Recognized Text (English): {recognized_text}")

                # Translate the English text to Cantonese using M2M100
                cantonese_translation = cls.translate_text_with_m2m100(
                    recognized_text, "en", "zh"
                )

                print(f"Translated Text (Cantonese): {cantonese_translation}")

                # Speak out the Cantonese translation
                cls.speakText(cantonese_translation)

                return recognized_text, cantonese_translation

            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return "", ""
            except sr.UnknownValueError:
                print("Unknown error occurred.")
                return "", ""

    @classmethod
    def translate_text_with_m2m100(cls, text, source_lang, target_lang):
        """
        Function to translate text using M2M100 model.

        Args:
            text (str): Text to be translated.
            source_lang (str): Source language code.
            target_lang (str): Target language code.

        Returns:
            str: Translated text.
        """
        translation_model_name = "facebook/m2m100_418M"
        translation_tokenizer = M2M100Tokenizer.from_pretrained(translation_model_name)
        translation_model = M2M100ForConditionalGeneration.from_pretrained(
            translation_model_name
        )
        # Set source and target languages for M2M100
        translation_tokenizer.src_lang = source_lang
        encoded_text = translation_tokenizer(text, return_tensors="pt")

        # Generate translation with forced BOS token for target language
        generated_tokens = translation_model.generate(
            **encoded_text,
            forced_bos_token_id=translation_tokenizer.get_lang_id(target_lang),
        )
        translated_text = translation_tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]
        return translated_text
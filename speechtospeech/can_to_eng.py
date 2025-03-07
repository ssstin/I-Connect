import speech_recognition as sr
from gtts import gTTS
import torch
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    MarianMTModel,
    MarianTokenizer,
)
from pydub import AudioSegment
import numpy as np
import os
import tempfile
from playsound import playsound


class Can_to_eng(object):
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Load the Wav2Vec 2.0 model and processor for Cantonese
    model_name = "CAiRE/wav2vec2-large-xlsr-53-cantonese"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    # Load the translation model and tokenizer for Cantonese (zh) to English (en)
    translation_model_name = "Helsinki-NLP/opus-mt-zh-en"
    translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    translation_model = MarianMTModel.from_pretrained(translation_model_name)

    @classmethod
    def SpeakText(cls, command):
        """
        Function to convert text to speech using gTTS.

        Args:
            command (str): The text to be spoken.
        """
        print(f"Speaking in English: {command}")
        
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_filename = temp_file.name
            
        # Generate the audio file with gTTS - use English
        tts = gTTS(text=command, lang='en')
        tts.save(temp_filename)
        
        # Play the audio file
        playsound(temp_filename)
        
        # Clean up the file after playing
        try:
            os.unlink(temp_filename)
        except:
            pass  # File may be in use, will be cleaned up later

    @classmethod
    def translate_cantonese_to_english_speech(cls):
        """
        Function to recognize speech in Cantonese and translate it to English speech.
        """
        with sr.Microphone() as source:
            # Wait for a second to adjust the recognizer to the ambient noise level
            print("Listening in Cantonese...")
            cls.recognizer.adjust_for_ambient_noise(source, duration=0.5)

            # Listen for the user's input
            audio = cls.recognizer.listen(source)

            try:
                # Convert the audio to a format suitable for the model using PyDub
                audio_data = np.frombuffer(audio.get_raw_data(), np.int16)
                audio_segment = AudioSegment(
                    data=audio_data.tobytes(),
                    sample_width=audio.sample_width,
                    frame_rate=audio.sample_rate,
                    channels=1,
                )

                # Ensure the audio is in the correct format (16kHz, mono)
                audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                samples = np.array(audio_segment.get_array_of_samples())

                # Convert the samples to float32 for the model
                audio_input = (
                    torch.tensor(samples, dtype=torch.float32) / 32768.0
                )  # Normalizing to range [-1, 1]

                # Use the processor to prepare the inputs correctly
                inputs = cls.processor(
                    audio_input, sampling_rate=16000, return_tensors="pt", padding=True
                )

                # Perform ASR
                with torch.no_grad():
                    logits = cls.model(
                        inputs.input_values, attention_mask=inputs.attention_mask
                    ).logits

                # Decode the logits to get the transcription in Cantonese
                predicted_ids = torch.argmax(logits, dim=-1)
                cantonese_transcription = cls.processor.batch_decode(predicted_ids)[0]

                print(f"Transcribed Text (Cantonese): {cantonese_transcription}")

                # Translate the Cantonese text to English using MarianMT
                if (
                    cantonese_transcription.strip()
                ):  # Check if the transcription is not empty
                    translated = cls.translate_text(cantonese_transcription)
                    print(f"Translated Text (English): {translated}")
                    cls.SpeakText(translated)
                else:
                    print("No valid transcription received.")

            except Exception as e:
                print(f"An error occurred during speech recognition: {e}")

    @classmethod
    def translate_cantonese_to_english_speech_with_return(cls):
        """
        Function to recognize speech in Cantonese, translate it to English speech,
        and return both the original and translated texts.
        
        Returns:
            tuple: (cantonese_text, english_translation)
        """
        with sr.Microphone() as source:
            # Wait for a second to adjust the recognizer to the ambient noise level
            print("Listening in Cantonese...")
            cls.recognizer.adjust_for_ambient_noise(source, duration=0.5)

            # Listen for the user's input
            audio = cls.recognizer.listen(source)

            try:
                # Convert the audio to a format suitable for the model using PyDub
                audio_data = np.frombuffer(audio.get_raw_data(), np.int16)
                audio_segment = AudioSegment(
                    data=audio_data.tobytes(),
                    sample_width=audio.sample_width,
                    frame_rate=audio.sample_rate,
                    channels=1,
                )

                # Ensure the audio is in the correct format (16kHz, mono)
                audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                samples = np.array(audio_segment.get_array_of_samples())

                # Convert the samples to float32 for the model
                audio_input = (
                    torch.tensor(samples, dtype=torch.float32) / 32768.0
                )  # Normalizing to range [-1, 1]

                # Use the processor to prepare the inputs correctly
                inputs = cls.processor(
                    audio_input, sampling_rate=16000, return_tensors="pt", padding=True
                )

                # Perform ASR
                with torch.no_grad():
                    logits = cls.model(
                        inputs.input_values, attention_mask=inputs.attention_mask
                    ).logits

                # Decode the logits to get the transcription in Cantonese
                predicted_ids = torch.argmax(logits, dim=-1)
                cantonese_transcription = cls.processor.batch_decode(predicted_ids)[0]

                print(f"Transcribed Text (Cantonese): {cantonese_transcription}")

                # Translate the Cantonese text to English using MarianMT
                if cantonese_transcription.strip():  # Check if the transcription is not empty
                    translated = cls.translate_text(cantonese_transcription)
                    print(f"Translated Text (English): {translated}")
                    # Speak out the English translation
                    cls.SpeakText(translated)
                    return cantonese_transcription, translated
                else:
                    print("No valid transcription received.")
                    return "", ""

            except Exception as e:
                print(f"An error occurred during speech recognition: {e}")
                return "", ""

    @classmethod
    def translate_text(cls, cantonese_text):
        """
        Function to translate Cantonese text to English.
        """
        # Encode the Cantonese text for the translation model
        translated = cls.translation_tokenizer(
            cantonese_text, return_tensors="pt", padding=True
        )
        # Perform the translation
        generated_tokens = cls.translation_model.generate(**translated)
        # Decode the translated tokens to get the English translation
        translation = cls.translation_tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]
        return translation
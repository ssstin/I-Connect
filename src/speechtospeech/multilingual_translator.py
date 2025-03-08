import speech_recognition as sr
from gtts import gTTS
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from playsound import playsound
import os
import tempfile

class MultilingualTranslator:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.model_name = "facebook/m2m100_418M"
        self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_name)

    def speak_text(self, text, lang_code):
        """Convert text to speech in the specified language."""
        print(f"Speaking in {lang_code}: {text}")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_filename = temp_file.name
        
        # Handle language codes for gTTS (some languages may need mapping)
        gtts_lang = lang_code.split('-')[0] if '-' in lang_code else lang_code
        try:
            tts = gTTS(text=text, lang=gtts_lang)
            tts.save(temp_filename)
        except ValueError:
            # Fallback to English if language not supported
            print(f"Language {gtts_lang} not supported by gTTS, falling back to English")
            tts = gTTS(text=text, lang='en')
            tts.save(temp_filename)
        
        playsound(temp_filename)
        try:
            os.unlink(temp_filename)
        except:
            pass

    def translate_text(self, text, source_lang, target_lang):
        """Translate text using M2M100 model."""
        self.tokenizer.src_lang = source_lang
        encoded_text = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(
            **encoded_text,
            forced_bos_token_id=self.tokenizer.get_lang_id(target_lang)
        )
        translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated_text

    def translate_speech(self, source_lang, target_lang):
        """Recognize speech and translate it to the target language."""
        with sr.Microphone() as source:
            print(f"Listening in {source_lang}...")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = self.recognizer.listen(source)

            try:
                # Recognize speech using Google Speech Recognition
                recognized_text = self.recognizer.recognize_google(audio, language=source_lang)
                print(f"Recognized Text ({source_lang}): {recognized_text}")

                # Translate to target language
                translated_text = self.translate_text(recognized_text, source_lang, target_lang)
                print(f"Translated Text ({target_lang}): {translated_text}")

                # Speak the translated text
                self.speak_text(translated_text, target_lang)

                return recognized_text, translated_text

            except sr.UnknownValueError:
                print("Could not understand the audio")
                return "", ""
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return "", ""
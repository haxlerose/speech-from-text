from transformers import AutoProcessor, AutoModel
import scipy.io.wavfile
from flask import Flask, request
import numpy as np
import time

app = Flask(__name__)

TTS_MODEL = "suno/bark-small"
VOICE_PRESET_DEFAULT = "v2/en_speaker_6"
VOICE_GENDER = 'MAN'

def voice_preset():
    voice_preset = request.json.get('voice_preset', None)
    if voice_preset is not None and isinstance(voice_preset, int) and voice_preset >= 0:
        voice_preset = "v2/en_speaker_" + str(voice_preset)
    else:
        voice_preset = VOICE_PRESET_DEFAULT
    return voice_preset

def process_text(text, voice_preset):
    processor = AutoProcessor.from_pretrained(TTS_MODEL)
    model = AutoModel.from_pretrained(TTS_MODEL)
    text = "[" + VOICE_GENDER + "]" + " " + "[speed: 1.0] [volume: 1.0]" + text
    inputs = processor(text, return_tensors="pt", voice_preset=voice_preset)
    return model, inputs

def generate_speech(model, inputs):
    speech_values = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], do_sample=True, pad_token_id=10000)
    audio = speech_values.detach().numpy().squeeze()
    audio = np.clip(audio, -1, 1)
    audio = np.int16(audio * 32767)
    return audio

def write_output(filename, audio):
    scipy.io.wavfile.write(filename, 24000, audio)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    start_time = time.time()

    voice = voice_preset()
    text = request.json['text']
    model, inputs = process_text(text, voice)
    audio = generate_speech(model, inputs)

    filename = f"{TTS_MODEL}_{voice}_{VOICE_GENDER}_speed_vol_output.wav".replace("/", "_")
    write_output(filename, audio)

    processing_time = time.time() - start_time
    return {"message": "Speech synthesized successfully", "text": text, "processing_time_sec": processing_time, "processing_time_min": processing_time/60}

if __name__ == '__main__':
    app.run(debug=True)

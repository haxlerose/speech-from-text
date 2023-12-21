from transformers import AutoProcessor, AutoModel
import scipy.io.wavfile
from flask import Flask, request
import numpy as np
import time

app = Flask(__name__)

TTS_MODEL = "suno/bark-small"
VOICE_PRESET_DEFAULT = "v2/en_speaker_6"
voice_gender = 'MAN'

@app.route('/synthesize', methods=['POST'])
def synthesize():
    start_time = time.time()

    processor = AutoProcessor.from_pretrained(TTS_MODEL)
    model = AutoModel.from_pretrained(TTS_MODEL)

    text = request.json['text']
    text = "[" + voice_gender + "]" + " " + text

    voice_preset = request.json.get('voice_preset', None)
    if voice_preset is not None and isinstance(voice_preset, int) and voice_preset >= 0:
        voice_preset = "v2/en_speaker_" + str(voice_preset)
    else:
        voice_preset = VOICE_PRESET_DEFAULT

    inputs = processor(text, return_tensors="pt", voice_preset=voice_preset)
    speech_values = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], do_sample=True, pad_token_id=10000)
    audio = speech_values.detach().numpy().squeeze()
    audio = np.clip(audio, -1, 1)
    audio = np.int16(audio * 32767)

    filename = f"{TTS_MODEL}_{voice_preset}_{voice_gender}_output.wav".replace("/", "_")
    scipy.io.wavfile.write(filename, 24000, audio)

    processing_time = time.time() - start_time
    return {"message": "Speech synthesized successfully", "text": text, "processing_time_sec": processing_time, "processing_time_min": processing_time/60}

if __name__ == '__main__':
    app.run(debug=True)

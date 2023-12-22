from dotenv import load_dotenv
from flask import Flask, request
from transformers import AutoProcessor, AutoModel
import numpy as np
import boto3
import logging
import os
import scipy.io.wavfile
import time

load_dotenv()

# to make an API call to this server, run the following command in a terminal:
# curl -X POST -H "Content-Type: application/json" -d '{"text":"Hello World!", "voice_preset":4}' YOUR_URL_HERE/synthesize

logging.basicConfig(level=logging.INFO)

FIXED_MODEL = True  # Set this to False to allow the client to specify the model
TTS_DEFAULT_MODEL = "suno/bark-small" # Used if FIXED_MODEL is True or no client model. Larger option: "suno/bark"
DEFAULT_TEXT_SPEED = 1.0 # Accepts a range from 0.5 to 2.0
DEFAULT_TEXT_VOLUME = 1.0 # Accepts a range from 0.5 to 2.0
VOICE_GENDER = 'MAN'
VOICE_PRESET_DEFAULT = "6" # Uses only libraries begining with 'v2/en_speaker_' from suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683
PAD_TOKEN_ID = 10000
SAMPLE_RATE = 24000
USE_AWS_S3 = True

app = Flask(__name__)

if USE_AWS_S3:
    assert os.getenv('AWS_ACCESS_KEY_ID') is not None, "AWS_ACCESS_KEY_ID must be set if USE_AWS_S3 is True"
    assert os.getenv('AWS_SECRET_ACCESS_KEY') is not None, "AWS_SECRET_ACCESS_KEY must be set if USE_AWS_S3 is True"
    assert os.getenv('S3_BUCKET_NAME') is not None, "S3_BUCKET_NAME must be set if USE_AWS_S3 is True"

if FIXED_MODEL:
    logging.info("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(TTS_DEFAULT_MODEL)
    model = AutoModel.from_pretrained(TTS_DEFAULT_MODEL)
    logging.info("Model and processor loaded.")


def get_voice_preset(request_json):
    """Extracts the voice preset from the request JSON. Defaults to VOICE_PRESET_DEFAULT if not present."""
    voice_preset = request_json.get('voice_preset', VOICE_PRESET_DEFAULT)
    if not (isinstance(voice_preset, int) and voice_preset >= 0):
        voice_preset = VOICE_PRESET_DEFAULT
    return "v2/en_speaker_" + str(voice_preset)

def get_model(request_json):
    """Extracts the model from the request JSON. Defaults to TTS_DEFAULT_MODEL if not present."""
    global model, processor
    if FIXED_MODEL:
        return model, processor
    else:
        model_name = request_json.get('model', TTS_DEFAULT_MODEL)
        return AutoModel.from_pretrained(model_name), AutoProcessor.from_pretrained(model_name)

def process_text(text, voice_preset, model, processor):
    """Processes the text and returns the inputs."""
    inputs = processor(text, return_tensors="pt", voice_preset=voice_preset)
    return inputs

def generate_speech(model, inputs):
    """Generates speech from the model and inputs."""
    speech_values = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], do_sample=True, pad_token_id=PAD_TOKEN_ID)
    audio = speech_values.detach().numpy().squeeze()
    audio = np.clip(audio, -1, 1)
    audio = np.int16(audio * 32767)
    return audio

def write_output(filename, audio):
    """Writes the audio to a file and uploads to AWS S3."""
    scipy.io.wavfile.write(filename, SAMPLE_RATE, audio)
    if USE_AWS_S3:
        try:
            s3 = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
            bucket_name = os.getenv('S3_BUCKET_NAME')
            s3.upload_file(filename, bucket_name, filename)
        except Exception as e:
            logging.error(f"Failed to upload file to S3: {e}")
        finally:
            os.remove(filename)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """Handles the /synthesize route."""
    start_time = time.time()

    text = request.json.get('text')
    if text is None:
        return {"message": "No text provided", "status": "error"}, 400
    else:
        text = f"[{VOICE_GENDER}] [speed: {DEFAULT_TEXT_SPEED}] [volume: {DEFAULT_TEXT_VOLUME}] {text}"

    voice = get_voice_preset(request.json)

    model, processor = get_model(request.json)
    inputs = process_text(text, voice, model, processor)
    audio = generate_speech(model, inputs)

    filename = (
        f"{model.name_or_path}_{voice}_{VOICE_GENDER}"
        f"_speed_{DEFAULT_TEXT_SPEED:.1f}"
        f"_vol_{DEFAULT_TEXT_VOLUME:.1f}"
    ).replace(".", "_").replace("/", "_") + ".wav"
    write_output(filename, audio)

    processing_time = time.time() - start_time
    return {
        "message": "Speech synthesized successfully",
        "filename": filename,
        "text": text,
        "processing_time_sec": processing_time,
        "processing_time_min": processing_time/60
        }

if __name__ == '__main__':
    app.run(debug=True)

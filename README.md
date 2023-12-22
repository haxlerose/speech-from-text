# Text-to-Speech Synthesis API

This repository contains a Flask API for a text-to-speech synthesis application. The application uses the Bark model created by Suno from Hugging Face's Transformers library to generate speech from text.

The API has a single endpoint, `/synthesize`, which accepts a POST request with a JSON body containing the following fields:

- `text`: The text to be synthesized into speech.
- `voice_preset` (optional): An integer representing the voice preset to be used. If not provided, the default voice preset will be used.
- `model` (optional): The name of the model to be used. If not provided, the default model will be used.

The response from the API will be a JSON object containing the following fields:

- `message`: A message indicating the status of the request.
- `filename`: The filename the audio was saved as.
- `text`: The original text that was synthesized.
- `processing_time_sec`: The time it took to process the request in seconds.
- `processing_time_min`: The time it took to process the request in minutes.


## Bark

Bark Docs: https://huggingface.co/suno/bark
Bark Speaker Library: https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683
Transformaers Library Docs: https://github.com/huggingface/transformers

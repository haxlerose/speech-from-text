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


# Bark Documentation

Bark is a transformer-based text-to-audio model created by Suno. Bark can generate highly realistic, multilingual speech as well as other audio - including music, background noise and simple sound effects. The model can also produce nonverbal communications like laughing, sighing and crying. To support the research community, we are providing access to pretrained model checkpoints ready for inference.

The original github repo and model card can be found here.

This model is meant for research purposes only. The model output is not censored and the authors do not endorse the opinions in the generated content. Use at your own risk.

Two checkpoints are released:

- small (this checkpoint)
- large

## 🤗 Transformers Usage

You can run Bark locally with the 🤗 Transformers library from version 4.31.0 onwards.

1. First install the 🤗 Transformers library and scipy:
```bash
pip install --upgrade pip
pip install --upgrade transformers scipy
```

2. Run inference via the Text-to-Speech (TTS) pipeline. You can infer the bark model via the TTS pipeline in just a few lines of code!

```python
from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-speech", "suno/bark-small")

speech = synthesiser("Hello, my dog is cooler than you!", forward_params={"do_sample": True})

scipy.io.wavfile.write("bark_out.wav", rate=speech["sampling_rate"], data=speech["audio"])
```

3. Run inference via the Transformers modelling code. You can use the processor + generate code to convert text into a mono 24 kHz speech waveform for more fine-grained control.
```python
from transformers import AutoProcessor, AutoModel

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = AutoModel.from_pretrained("suno/bark-small")

inputs = processor(
    text=["Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe."],
    return_tensors="pt",
)

speech_values = model.generate(**inputs, do_sample=True)
```

4. Listen to the speech samples either in an ipynb notebook:
```python
from IPython.display import Audio

sampling_rate = model.generation_config.sample_rate
Audio(speech_values.cpu().numpy().squeeze(), rate=sampling_rate)
```

Or save them as a .wav file using a third-party library, e.g. scipy:

```python
import scipy

sampling_rate = model.config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=sampling_rate, data=speech_values.cpu().numpy().squeeze())
```

For more details on using the Bark model for inference using the 🤗 Transformers library, refer to the Bark docs.

### Optimization tips

Refers to this blog post to find out more about the following methods and a benchmark of their benefits.

### Get significant speed-ups:

#### Using 🤗 Better Transformer

Better Transformer is an 🤗 Optimum feature that performs kernel fusion under the hood. You can gain 20% to 30% in speed with zero performance degradation. It only requires one line of code to export the model to 🤗 Better Transformer:

```python
model =  model.to_bettertransformer()
```

Note that 🤗 Optimum must be installed before using this feature. Here's how to install it.

#### Using Flash Attention 2

Flash Attention 2 is an even faster, optimized version of the previous optimization.

```python
model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16, use_flash_attention_2=True).to(device)
```

Make sure to load your model in half-precision (e.g. `torch.float16``) and to install the latest version of Flash Attention 2.

**Note:** Flash Attention 2 is only available on newer GPUs, refer to 🤗 Better Transformer in case your GPU don't support it.

#### Reduce memory footprint:

##### Using half-precision

You can speed up inference and reduce memory footprint by 50% simply by loading the model in half-precision (e.g. `torch.float16``).

##### Using CPU offload

Bark is made up of 4 sub-models, which are called up sequentially during audio generation. In other words, while one sub-model is in use, the other sub-models are idle.

If you're using a CUDA device, a simple solution to benefit from an 80% reduction in memory footprint is to offload the GPU's submodels when they're idle. This operation is called CPU offloading. You can use it with one line of code.

```python
model.enable_cpu_offload()
```

Note that 🤗 Accelerate must be installed before using this feature. Here's how to install it.

#### Suno Usage

You can also run Bark locally through the original Bark library:

1. First install the bark library

2. Run the following Python code:

```python
from bark import SAMPLE_RATE, generate_audio, preload_models
from IPython.display import Audio

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
     Hello, my name is Suno. And, uh — and I like pizza. [laughs]
     But I also have other interests such as playing tic tac toe.
"""
speech_array = generate_audio(text_prompt)

# play text in notebook
Audio(speech_array, rate=SAMPLE_RATE)
```

To save audio_array as a WAV file:

```python
from scipy.io.wavfile import write as write_wav

write_wav("/path/to/audio.wav", SAMPLE_RATE, audio_array)
```

### Model Details

The following is additional information about the models released here.

Bark is a series of three transformer models that turn text into audio.

#### Text to semantic tokens
- Input: text, tokenized with BERT tokenizer from Hugging Face
- Output: semantic tokens that encode the audio to be generated

#### Semantic to coarse tokens
- Input: semantic tokens
- Output: tokens from the first two codebooks of the EnCodec Codec from facebook

#### Coarse to fine tokens
- Input: the first two codebooks from EnCodec
- Output: 8 codebooks from EnCodec

#### Architecture
| Model | Parameters | Attention | Output Vocab size |
|-------|------------|-----------|-------------------|
| Text to semantic tokens | 80/300 M | Causal | 10,000 |
| Semantic to coarse tokens | 80/300 M | Causal | 2x 1,024 |
| Coarse to fine tokens | 80/300 M | Non-causal | 6x 1,024 |

#### Release date

April 2023

### Broader Implications

We anticipate that this model's text to audio capabilities can be used to improve accessbility tools in a variety of languages.

While we hope that this release will enable users to express their creativity and build applications that are a force for good, we acknowledge that any text to audio model has the potential for dual use. While it is not straightforward to voice clone known people with Bark, it can still be used for nefarious purposes. To further reduce the chances of unintended use of Bark, we also release a simple classifier to detect Bark-generated audio with high accuracy (see notebooks section of the main repository).

### License

Bark is licensed under the MIT License, meaning it's available for commercial use.

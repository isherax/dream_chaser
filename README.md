# Dream Chaser Backend Server

A backend server for multilevel AI imaging.

## Setup

1. Download pretrained text to image model from huggingface in subdirectory: `git clone https://huggingface.co/prompthero/openjourney-v4`
2. Download pretrained prompt generation model from huggingface in subdirectory: `git clone https://huggingface.co/FredZhang7/distilgpt2-stable-diffusion-v2`
3. Download  tokenizer from huggingface in subdirectory: `git clone https://huggingface.co/distilgpt2`
3. Download pretrained sentence similarity model from huggingface in subdirectory: `git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2`

## Run

1. Execute command: `python start_api.py`

# API Requests

1. `\image`: Generate an image using the AI text-to-image model. Returns a png object in bytes form.
    1. `prompt` (required): Input text for image prompt.
    2. `height`:
    3. `width`:
    4. `num_inference_steps`:
    5. `guidance_scale`:
    6. `negative_prompt`:
    7. `num_images_per_prompt`:
2. `\text`: Improve a prompt using the AI text generation model. Returns a string.
    1. `prompt` (required):
    2. `temperature`:
    3. `top_k`:
    4. `max_length`:
    5. `repetition_penalty`:
    6. `num_return_sequences`:

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
    2. `enhance_prompt` (optional): Boolean variable to improve prompt using prompt generation model. Set to true by default.

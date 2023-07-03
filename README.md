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

1. `\image_gen`: Generate an image using the AI text-to-image model. Returns a png object in bytes form.
    1. `prompt` (required): input text for image generation model
    2. `height`: height of image in pixels
    3. `width`: width of image in pixels
    4. `num_inference_steps`: number of denoising steps
    5. `guidance_scale`: guidance scale
    6. `negative_prompt`: prompts to avoid in image generation
    7. `num_images_per_prompt`: number of images to return
2. `\image_upscale`: Upscale an image using the AI upscaler model. Returns a png object in bytes form.
    1. `url` (required): location of input image for upscaler model 
    2. `prompt` (required): input text for upscaler model
    3. `num_inference_steps`
    4. `guidance_scale`
    5. `negative_prompt`
    6. `num_images_per_prompt`
3. `\text_gen`: Improve a prompt using the AI text generation model. Returns a string.
    1. `prompt` (required): input text for text generation model
    2. `temperature`: diversity of prompt results
    3. `top_k`: number of tokens to sample at each step
    4. `max_length`: maximum number of output tokens
    5. `repetition_penalty`: penalty value for token repetition
    6. `num_return_sequences`: number of prompts to return

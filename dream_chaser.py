from diffusers import DiffusionPipeline
import argparse
import torch
import transformers
import uuid
import warnings


class DreamChaserAPI:
    def __init__(self, image_model_id, text_model_id):
        self.image_pipeline = DiffusionPipeline.from_pretrained(image_model_id)
        self.text_pipeline = transformers.pipeline('text-generation', model=text_model_id)
        
        if torch.cuda.device_count() > 0:
            self.image_pipeline.to('cuda')
        else:
            warnings.warn('GPU not found, image generation will be substantially slower.', RuntimeWarning)

            
    def improve_prompt(self, text):
        sequences = self.text_pipeline(text, max_length=77)
        
        return sequences[0]['generated_text']


if __name__ == '__main__':
    image_model = './openjourney'
    text_model = './gpt2-650k-stable-diffusion-prompt-generator'
    api = DreamChaserAPI(image_model, text_model)
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', '--prompt', required=True)
    args = parser.parse_args()
    
    new_prompt = api.improve_prompt(args.p)
    print(new_prompt)
    images = api.image_pipeline(new_prompt).images
    
    for i in range(len(images)):
        temp_file = uuid.uuid4().hex
        images[i].save(f'samples/{temp_file}.png')

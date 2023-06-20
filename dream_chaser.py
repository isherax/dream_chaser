from diffusers import DiffusionPipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse
import torch
import uuid
import warnings


class DreamChaserAPI:
    def __init__(self, image_model_id, text_model_id):
        self.image_pipeline = DiffusionPipeline.from_pretrained(image_model_id)
        self.tokenizer = T5Tokenizer.from_pretrained(text_model_id)
        
        if torch.cuda.device_count() > 0:
            self.has_gpu = True
            self.image_pipeline.to('cuda')
            self.text_model = T5ForConditionalGeneration.from_pretrained(text_model_id)
        else:
            warnings.warn('GPU not found, image generation will be substantially slower.', RuntimeWarning)
            self.has_gpu = False
            self.text_model = T5ForConditionalGeneration.from_pretrained(text_model_id, device_map='auto')
            
            
    def improve_prompt(self, text):
        instructions = 'An ideal prompt for a text to image model should be concise, clear, and describe the artistic style of the image. Expand or summarize the following prompt as necessary to retain the original meaning while optimizing for good image generation'
        input_text = f'{instructions}: {text}'
        
        if self.has_gpu:
            input_ids = self.tokenizer(input_text, return_tensors='pt').input_ids.to('cuda')
        else:
            input_ids = self.tokenizer(input_text, return_tensors='pt').input_ids
            
        output_tokens = self.text_model.generate(input_ids)
        output_text = self.tokenizer.decode(output_tokens[0])
        
        return output_text


if __name__ == '__main__':
    image_model = './openjourney'
    text_model = './flan-t5-base'
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

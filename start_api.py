from diffusers import StableDiffusionPipeline
from flask import Flask, send_file
from flask_restful import Resource, Api, reqparse
import io
import torch
import transformers
import warnings


class DreamChaserAPI(Resource):
    def __init__(self):
        self.__name__ = 'DreamChaserAPI'
        
            
    @classmethod
    def setup_models(self, image_model_id, text_model_id):
        self.image_pipeline = StableDiffusionPipeline.from_pretrained(image_model_id) 
        self.text_pipeline = transformers.pipeline('text-generation', model=text_model_id)
        
        if torch.cuda.device_count() > 0:
            self.image_pipeline.to('cuda')
        else:
            warnings.warn('GPU not found, image generation will be substantially slower.', RuntimeWarning)
            
        return self
    
    
    def enhance_prompt(self, input_text):
        text = self.text_pipeline(input_text, max_length=80)[0]['generated_text']
        text = ' '.join(text.split(' ')[:-1 or None])
        
        return text
            
            
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('prompt', required=True)
        parser.add_argument('enhance_prompt', default=True)
        args = parser.parse_args()
        
        if args['enhance_prompt']:
            final_prompt = self.enhance_prompt(args['prompt'])
        else:
            final_prompt = args['prompt']
        
        image = self.image_pipeline(final_prompt).images[0]
        image_object = io.BytesIO()
        image.save(image_object, 'PNG')
        image_object.seek(0)
        
        return send_file(image_object, mimetype='image/PNG')
    

if __name__ == '__main__':
    image_model = './openjourney-v4'
    text_model = './gpt2-650k-stable-diffusion-prompt-generator'
    dca = DreamChaserAPI.setup_models(image_model, text_model)
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(dca, '/image')
    app.run()
    
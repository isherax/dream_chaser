from diffusers import StableDiffusionPipeline
from flask import Flask
from flask_restful import Resource, Api, reqparse
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
            
            
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('prompt', required=True)
        args = parser.parse_args()
        images = self.image_pipeline(args['prompt']).images
        
        return images, 200
    

if __name__ == '__main__':
    image_model = './openjourney-v4'
    text_model = './gpt2-650k-stable-diffusion-prompt-generator'
    dca = DreamChaserAPI.setup_models(image_model, text_model)
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(dca, '/image')
    app.run()
    
from diffusers import StableDiffusionPipeline
from flask import send_file
from flask_restful import Resource, reqparse
from io import BytesIO
from warnings import warn
import requests
import torch


class ImageGen(Resource):
    def __init__(self):
        self.__name__ = 'DreamChaserImageAPI'
        self.text_url = 'http://127.0.0.1:5000/text_gen'
        
            
    @classmethod
    def setup_models(self, image_model_id):
        self.image_pipeline = StableDiffusionPipeline.from_pretrained(image_model_id) 
        
        if torch.cuda.device_count() > 0:
            self.image_pipeline.to('cuda')
        else:
            warn('GPU not found, image generation will be substantially slower.', RuntimeWarning)
            
        return self
            
            
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('prompt', type=str, required=True, location='args')
        parser.add_argument('enhance', type=bool, default=True, location='args')
        parser.add_argument('height', type=int, default=512, location='args')
        parser.add_argument('width', type=int, default=512, location='args')
        parser.add_argument('num_inference_steps', type=int, default=50, location='args')
        parser.add_argument('guidance_scale', type=float, default=7.5, location='args')
        parser.add_argument('negative_prompt', type=str, default='', location='args')
        parser.add_argument('num_images_per_prompt', type=int, default=1, location='args')
        args = parser.parse_args()
        
        if args['enhance']:
            get_request = requests.get(self.text_url, params={'prompt': args['prompt']})
            final_prompt = get_request.text
        else:
            final_prompt = args['prompt']
        
        image = self.image_pipeline(prompt=final_prompt,
                                    height=args['height'],
                                    width=args['width'],
                                    num_inference_steps=args['num_inference_steps'],
                                    guidance_scale=args['guidance_scale'],
                                    negative_prompt=args['negative_prompt'],
                                    num_images_per_prompt=args['num_images_per_prompt']).images[0]
        image_object = BytesIO()
        image.save(image_object, 'PNG')
        image_object.seek(0)
        
        return send_file(image_object, mimetype='image/PNG')
    
from diffusers import StableDiffusionUpscalePipeline
from flask import send_file
from flask_restful import Resource, reqparse
from io import BytesIO
from PIL import Image
from warnings import warn
import requests
import torch


class ImageUpscaler(Resource):
    def __init__(self):
        self.__name__ = 'DreamChaserUpscalerAPI'
        
            
    @classmethod
    def setup_models(self, upscaler_id):
        self.upscaler_pipeline = StableDiffusionUpscalePipeline.from_pretrained(upscaler_id) 
        
        if torch.cuda.device_count() > 0:
            self.upscaler_pipeline.to('cuda')
        else:
            warn('GPU not found, image generation will be substantially slower.', RuntimeWarning)
            
        return self

    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('url', type=str, required=True, location='args')
        parser.add_argument('prompt', type=str, required=True, location='args')
        parser.add_argument('num_inference_steps', type=int, default=50, location='args')
        parser.add_argument('guidance_scale', type=float, default=7.5, location='args')
        parser.add_argument('negative_prompt', type=str, default='', location='args')
        parser.add_argument('num_images_per_prompt', type=int, default=1, location='args')
        args = parser.parse_args()
        
        response = requests.get(args['url'])
        original_image = Image.open(BytesIO(response.content)).convert('RGB')
        upscaled_image = self.upscaler_pipeline(prompt=args['prompt'], 
                                                image=original_image,
                                                num_inference_steps=args['num_inference_steps'],
                                                guidance_scale=args['guidance_scale'],
                                                negative_prompt=args['negative_prompt'],
                                                num_images_per_prompt=args['num_images_per_prompt']).images[0]
        image_object = BytesIO()
        upscaled_image.save(image_object, 'PNG')
        image_object.seek(0)
        
        return send_file(image_object, mimetype='image/PNG')
        
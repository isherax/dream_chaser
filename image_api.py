from diffusers import StableDiffusionPipeline
from flask import send_file
from flask_restful import Resource, reqparse
from io import BytesIO
from warnings import warn
import torch


class ImageAPI(Resource):
    def __init__(self):
        self.__name__ = 'DreamChaserAPI'
        
            
    @classmethod
    def setup_models(self, image_model_id):
        self.image_pipeline = StableDiffusionPipeline.from_pretrained(image_model_id) 
        
        if torch.cuda.device_count() > 0:
            self.image_pipeline.to('cuda')
        else:
            warn('GPU not found, image generation will be substantially slower.', RuntimeWarning)
            
        return self
            
            
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('prompt', required=True)
        args = parser.parse_args()   
        image = self.image_pipeline(args['prompt']).images[0]
        image_object = BytesIO()
        image.save(image_object, 'PNG')
        image_object.seek(0)
        
        return send_file(image_object, mimetype='image/PNG')
    
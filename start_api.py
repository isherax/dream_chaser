from diffusers import StableDiffusionPipeline
from flask import Flask, send_file
from flask_restful import Resource, Api, reqparse
from sentence_transformers import SentenceTransformer, util
import io
import torch
import transformers
import warnings


class DreamChaserAPI(Resource):
    def __init__(self):
        self.__name__ = 'DreamChaserAPI'
        
            
    @classmethod
    def setup_models(self, image_model_id, prompt_model_id, similarity_model_id):
        self.image_pipeline = StableDiffusionPipeline.from_pretrained(image_model_id) 
        self.prompt_pipeline = transformers.pipeline('text-generation', model=prompt_model_id)
        self.similarity_model = SentenceTransformer(similarity_model_id)
        
        if torch.cuda.device_count() > 0:
            self.image_pipeline.to('cuda')
            self.prompt_pipeline.to('cuda')
        else:
            warnings.warn('GPU not found, image generation will be substantially slower.', RuntimeWarning)
            
        return self
    
    
    def enhance_prompt(self, input_text, num_gens=5):
        input_embeddings = self.similarity_model.encode(input_text, convert_to_tensor=True)
        new_prompt = input_text
        max_similarity = 0
        
        for i in range(num_gens):
            text = self.prompt_pipeline(input_text, 
                                        max_length=77, 
                                        pad_token_id=self.prompt_pipeline.tokenizer.eos_token_id)[0]['generated_text']
            text = ' '.join(text.split(' ')[:-1 or None])
            new_embeddings = self.similarity_model.encode(text, convert_to_tensor=True)
            cos_score = util.cos_sim(input_embeddings, new_embeddings).numpy()[0][0]
            
            if cos_score > max_similarity:
                max_similarity = cos_score
                new_prompt = text
        
        return new_prompt
            
            
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
    image_model_folder = './openjourney-v4'
    text_model_folder = './gpt2-650k-stable-diffusion-prompt-generator'
    similarity_model_folder = './all-MiniLM-L6-v2'
    dca = DreamChaserAPI.setup_models(image_model_folder, text_model_folder, similarity_model_folder)
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(dca, '/image')
    app.run()
    
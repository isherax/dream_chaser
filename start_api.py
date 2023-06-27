from diffusers import StableDiffusionPipeline
from flask import Flask, send_file
from flask_restful import Resource, Api, reqparse
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import io
import torch
import warnings


class DreamChaserAPI(Resource):
    def __init__(self):
        self.__name__ = 'DreamChaserAPI'
        
            
    @classmethod
    def setup_models(self, image_model_id, prompt_model_id, prompt_tokenizer_id, similarity_model_id):
        self.image_pipeline = StableDiffusionPipeline.from_pretrained(image_model_id) 
        self.prompt_model = GPT2LMHeadModel.from_pretrained(prompt_model_id)
        self.prompt_tokenizer = GPT2Tokenizer.from_pretrained(prompt_tokenizer_id)
        self.prompt_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.similarity_model = SentenceTransformer(similarity_model_id)
        
        if torch.cuda.device_count() > 0:
            self.image_pipeline.to('cuda')
            self.prompt_pipeline.to('cuda')
        else:
            warnings.warn('GPU not found, image generation will be substantially slower.', RuntimeWarning)
            
        return self
    
    
    def enhance_prompt(self, input_text, temperature=0.9, top_k=8, max_length=80, repetition_penalty=1.2, num_return_sequences=5):
        best_prompt = input_text
        max_similarity = 0
        input_embeddings = self.similarity_model.encode(input_text, convert_to_tensor=True)
        input_ids = self.prompt_tokenizer(input_text, return_tensors='pt').input_ids
        new_prompts = self.prompt_model.generate(input_ids, 
                                                 do_sample=True, 
                                                 temperature=temperature, 
                                                 top_k=top_k, 
                                                 max_length=max_length, 
                                                 num_return_sequences=num_return_sequences, 
                                                 repetition_penalty=repetition_penalty, 
                                                 penalty_alpha=0.6, 
                                                 no_repeat_ngram_size=1, 
                                                 early_stopping=True)
        
        for i in range(len(new_prompts)):
            text = self.prompt_tokenizer.decode(new_prompts[i], skip_special_tokens=True)
            new_embeddings = self.similarity_model.encode(text, convert_to_tensor=True)
            cos_score = util.cos_sim(input_embeddings, new_embeddings).numpy()[0][0]
            
            if cos_score > max_similarity:
                max_similarity = cos_score
                best_prompt = text
        
        return best_prompt
            
            
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
    prompt_model_folder = './distilgpt2-stable-diffusion-v2'
    prompt_tokenizer_folder = './distilgpt2'
    similarity_model_folder = './all-MiniLM-L6-v2'
    dca = DreamChaserAPI.setup_models(image_model_folder, 
                                      prompt_model_folder,
                                      prompt_tokenizer_folder,
                                      similarity_model_folder)
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(dca, '/image')
    app.run()
    
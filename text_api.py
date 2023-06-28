from sentence_transformers import SentenceTransformer, util
from flask_restful import Resource, reqparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class TextAPI(Resource):
    def __init__(self):
        self.__name__ = 'DreamChaserTextAPI'
        
    @classmethod
    def setup_models(self, prompt_model_id, prompt_tokenizer_id, similarity_model_id):
        self.prompt_model = GPT2LMHeadModel.from_pretrained(prompt_model_id)
        self.prompt_tokenizer = GPT2Tokenizer.from_pretrained(prompt_tokenizer_id)
        self.prompt_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.similarity_model = SentenceTransformer(similarity_model_id)
        
        return self
    
        
    def enhance_prompt(self, input_text, temperature, top_k, max_length, repetition_penalty, num_return_sequences):
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
        
        # select best prompt based on semantic similarity to original prompt
        for new_prompt in new_prompts:
            text = self.prompt_tokenizer.decode(new_prompt, skip_special_tokens=True)
            new_embeddings = self.similarity_model.encode(text, convert_to_tensor=True)
            cos_score = util.cos_sim(input_embeddings, new_embeddings).numpy()[0][0]
            
            if cos_score > max_similarity:
                max_similarity = cos_score
                best_prompt = text
        
        return best_prompt
    
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('prompt', required=True)
        parser.add_argument('temperature', default=0.9)
        parser.add_argument('top_k', default=8)
        parser.add_argument('max_length', default=77)
        parser.add_argument('repetition_penalty', default=1.2)
        parser.add_argument('num_return_sequences', default=5)
        args = parser.parse_args()
        
        new_prompt = self.enhance_prompt(args['prompt'], 
                                         args['temperature'], 
                                         args['top_k'], 
                                         args['max_length'], 
                                         args['repetition_penalty'], 
                                         args['num_return_sequences'])
        
        return new_prompt
    
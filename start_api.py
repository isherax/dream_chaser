from flask import Flask
from flask_restful import Api
from resources.image_gen import ImageGen
from resources.text_gen import TextGen
    

if __name__ == '__main__':
    image_model_folder = './openjourney-v4'
    prompt_model_folder = './distilgpt2-stable-diffusion-v2'
    prompt_tokenizer_folder = './distilgpt2'
    similarity_model_folder = './all-MiniLM-L6-v2'
    image_generator = ImageGen.setup_models(image_model_folder)
    text_generator = TextGen.setup_models(prompt_model_folder, prompt_tokenizer_folder, similarity_model_folder)
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(image_generator, '/image')
    api.add_resource(text_generator, '/text')
    app.run()
    
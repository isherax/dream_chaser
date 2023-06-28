from flask import Flask
from flask_restful import Api
import image_api
import text_api
    

if __name__ == '__main__':
    image_model_folder = './openjourney-v4'
    prompt_model_folder = './distilgpt2-stable-diffusion-v2'
    prompt_tokenizer_folder = './distilgpt2'
    similarity_model_folder = './all-MiniLM-L6-v2'
    iapi = image_api.ImageAPI.setup_models(image_model_folder)
    tapi = text_api.TextAPI.setup_models(prompt_model_folder, prompt_tokenizer_folder, similarity_model_folder)
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(iapi, '/image')
    api.add_resource(tapi, '/text')
    app.run()
    
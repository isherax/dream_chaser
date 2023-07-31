from flask import Flask
from flask_restful import Api
from resources.image_gen import ImageGen
from resources.image_upscaler import ImageUpscaler
from resources.text_gen import TextGen
    

if __name__ == '__main__':
    image_gen_folder = './openjourney-v4'
    image_upscaler_folder= './stable-diffusion-x4-upscaler'
    prompt_gen_folder = './distilgpt2-stable-diffusion-v2'
    prompt_tokenizer_folder = './distilgpt2'
    similarity_model_folder = './all-MiniLM-L6-v2'
    image_generator = ImageGen.setup_models(image_gen_folder)
    image_upscaler = ImageUpscaler.setup_models(image_upscaler_folder)
    text_generator = TextGen.setup_models(prompt_gen_folder, prompt_tokenizer_folder, similarity_model_folder)
    app = Flask('DreamChaserAPI')
    api = Api(app)
    api.add_resource(image_generator, '/image_gen')
    api.add_resource(image_upscaler, '/image_upscaler')
    api.add_resource(text_generator, '/text_gen')
    
    with app.app_context():
        app.run()
    
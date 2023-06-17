from diffusers import DiffusionPipeline
import torch


class DreamChaserAPI:
    def __init__(self, model_id):
        self.pipeline = DiffusionPipeline.from_pretrained(model_id)
        
        # move generator to GPU if possible
        if torch.cuda.device_count() > 0:
            self.pipeline.to('cuda')


if __name__ == '__main__':
    pretrained_model = './stable-diffusion-v1-5'
    api = DreamChaserAPI(pretrained_model)
    image = api.pipeline('An image of a squirrel in Picasso style').images[0]
    image.save('sample.png')

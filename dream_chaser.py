from diffusers import DiffusionPipeline
import argparse
import torch
import uuid
import warnings


class DreamChaserAPI:
    def __init__(self, model_id):
        self.pipeline = DiffusionPipeline.from_pretrained(model_id)
        
        # move generator to GPU if possible
        if torch.cuda.device_count() > 0:
            self.pipeline.to('cuda')
        else:
            warnings.warn('GPU not found, image generation will be substantially slower.', RuntimeWarning)


if __name__ == '__main__':
    pretrained_model = './stable-diffusion-v1-5'
    api = DreamChaserAPI(pretrained_model)
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', '--prompt')
    args = parser.parse_args()
    images = api.pipeline(args.p).images
    
    for i in range(len(images)):
        temp_file = uuid.uuid4().hex
        images[i].save(f'samples/{temp_file}.png')

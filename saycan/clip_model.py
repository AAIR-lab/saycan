import saycan
import clip
import numpy as np

import torch

class CLIPModel:

    def __init__(self, model_dir=saycan.ASSETS_DIR,
        print_model_info=True):

        run_on_gpu = torch.cuda.is_available()
        device = "cuda" if run_on_gpu else "cpu"
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

        # Set the model in eval mode.
        if run_on_gpu:
            clip_model.cuda().eval()
        else:
            clip_model.eval()

        if print_model_info:
            print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
            print("Input resolution:", clip_model.visual.input_resolution)
            print("Context length:", clip_model.context_length)
            print("Vocab size:", clip_model.vocab_size)

        self.clip_model = clip_model

    def encode_text(self, text):
        return self.clip_model.encode_text(text)

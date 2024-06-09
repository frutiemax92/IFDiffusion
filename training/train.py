'''
'''

import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path
import sys
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

from models.discriminator import Discriminator
from models.generator import Generator
from transformers import T5Tokenizer, T5ForConditionalGeneration
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader, IterableDataset, Sampler
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from PIL import Image
from torchvision.transforms import Resize
from os import walk
import pathlib
import torch

class Bucket:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.image_paths = []
        self.caption_paths = []
    
    def get_length(self):
        return len(self.image_paths)

class BucketSampler(Sampler):
    def __init__(self, batch_size : int, buckets : list[Bucket]):
        self.batches = []

        # go through all the buckets and make batches
        for bucket in buckets:
            batch = []
            image_width = bucket.width
            image_height = bucket.height
            for b_length in range(bucket.get_length()):
                batch.append((str(bucket.image_paths[b_length]), str(bucket.caption_paths[b_length])))
                if len(batch) >= batch_size:
                    self.batches.append(batch)
                    batch = []
            if len(batch) != batch_size and len(batch) != 0:
                self.batches.append(batch)

    def __iter__(self):
        for batch in tqdm(self.batches, desc="#batch"):
            captions = []
            images = []
            for elem in batch:
                image_path, caption_path = elem
                with Image.open(image_path) as image:
                    with open(caption_path) as f:
                        caption = f.read()
                        image = pil_to_tensor(image)
                        captions.append(caption)
                        images.append(image)

            image_width = images[0].shape[2]
            image_height = images[0].shape[1]
            images_tensor = torch.zeros((len(batch), 3, image_height, image_width))
            for i in range(len(batch)):
                images_tensor[i] = images[i]
            yield (images_tensor, captions)

    def __len__(self):
        return len(self.batches)

class ImageDataset(Dataset):
    def find_bucket(self, width, height):
        return next((bucket for bucket in self.buckets if bucket.width == width and bucket.height == height), None)
    
    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self.buckets = []

        self.image_paths = []
        self.caption_paths = []
        for (dirpath, dirnames, filenames) in walk(folder):
            for filename in filenames:
                filepath = pathlib.Path(dirpath).joinpath(filename)
                suffix = pathlib.Path(filepath).suffix

                if suffix != '.png' and \
                    suffix != '.jpg' and \
                    suffix != '.webp' and \
                    suffix != '.jpeg':
                    continue
                
                # check if there is an associated .txt file
                filename = filepath.stem
                txt_file = filepath.parent.joinpath(filename + '.txt')
                if txt_file.exists():
                    # open the image to find its size
                    with Image.open(filepath) as image:
                        bucket = self.find_bucket(image.width, image.height)
                        if bucket == None:
                            self.buckets.append(Bucket(image.width, image.height))
                            bucket = self.buckets[-1]
                        
                        bucket.image_paths.append(filepath)
                        bucket.caption_paths.append(txt_file)

    def __getitem__(self, batch):
        # get the image size
        return batch

def train_loop(discriminator, original_images, batch_size, discriminator_features, discriminator_loss_fn,
               accelerator, optimizer, generator, ratios, embeddings, generator_loss_fn):
    # push the real image into the discriminator
    disciminator_pred = discriminator(original_images)
    real_preds = torch.ones((batch_size, discriminator_features)).to(device=disciminator_pred.device, dtype=disciminator_pred.dtype)
    loss = discriminator_loss_fn(disciminator_pred, real_preds)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

    # push a fake image into the discriminator
    fake_images = torch.rand_like(original_images)
    disciminator_pred = discriminator(fake_images)
    fake_preds = torch.zeros((batch_size, discriminator_features)).to(device=disciminator_pred.device, dtype=disciminator_pred.dtype)
    loss = discriminator_loss_fn(disciminator_pred, fake_preds)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    
    # generate an image
    generated_images = generator(fake_images, ratios, embeddings)
    disciminator_pred = discriminator(generated_images)

    # these should be identified close to real images!
    loss_d = discriminator_loss_fn(disciminator_pred, real_preds)
    loss_g = generator_loss_fn(generated_images, original_images)
    accelerator.backward(loss_d)
    accelerator.backward(loss_g)
    optimizer.step()
    optimizer.zero_grad()

def eval(generator, tokenizer : T5Tokenizer, encoder : T5ForConditionalGeneration, prompts,
        generator_embed_dim, width, height, device='cuda'):
    images = []

    with torch.no_grad():
        input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to(device=device)
        outputs = encoder.generate(input_ids, max_new_tokens=generator_embed_dim // 3).to(torch.float)

        # pad the outputs for the embedding dim/3
        embeddings = torch.zeros((outputs.shape[0], generator_embed_dim // 3))
        embeddings[:, :outputs.shape[1]] = outputs

        images = torch.rand((outputs.shape[0], 3, height, width))
        ratios = torch.full((outputs.shape[0],), height / width)

        embeddings = embeddings.to(device=device)
        images = images.to(device=device)
        embeddings = embeddings.to(device=device)
        images = generator(images, ratios, embeddings)

        # we need to resize the image for the ratio
        model_size = images.shape[2]
        new_width = model_size
        new_height = model_size
        if height < width:
            new_height = int(new_height * height / width)
        else:
            new_width = int(new_width * width / height)
        
        transform = Resize((new_height, new_width))
        images = transform(images)
        images = images.to(dtype=torch.uint8)

        i = 0
        for image in images:
            pil_image = to_pil_image(image)
            pil_image.save(f'test{i}.png')
            i = i + 1 
            
def train(args):
    dataset_folder = args.dataset_folder
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    steps_per_eval = args.steps_per_eval
    eval_prompts = args.eval_prompts

    dataset = ImageDataset(dataset_folder)
    bucket_sampler = BucketSampler(batch_size, dataset.buckets)

    # load the models
    image_size = args.image_size
    discriminator_features = args.discriminator_features
    generator_num_rows = args.generator_num_rows
    generator_num_heads = args.generator_num_heads
    generator_embed_dim = args.generator_embed_dim
    generator_ratio_mult = args.generator_ratio_mult
    eval_image_height = args.eval_image_height
    eval_image_width = args.eval_image_width
    discriminator = Discriminator(image_size, discriminator_features)
    generator = Generator(image_size, generator_num_rows, generator_num_heads, generator_embed_dim, generator_ratio_mult)

    # setup the optimizers
    optimizer = AdamW(params=generator.parameters(), lr=lr)
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
    encoder = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

    def collate_fn(batch):
        return batch[0]
    
    dataloader = DataLoader(dataset, sampler=bucket_sampler, collate_fn=collate_fn)

    # prepare the models
    accelerator = Accelerator(mixed_precision='bf16')
    
    dataloader, optimizer, tokenizer, encoder, discriminator, generator = \
        accelerator.prepare(dataloader, optimizer, tokenizer, encoder, discriminator, generator)

    # loss function for discriminator
    discriminator_loss_fn = torch.nn.BCELoss()

    # generated image loss for generator
    generator_loss_fn = torch.nn.MSELoss()

    # train loop
    step_count = 0
    for t in tqdm(range(epochs), desc='Epochs'):
        for batch in dataloader:
            original_images, captions = batch

            # calculate the ratios of the images
            image_width = original_images.shape[2]
            image_height = original_images.shape[1]
            batch_size = original_images.shape[0]
            ratios = torch.full((batch_size,), image_height / image_width)

            # rescale the original images down to a square image
            transform = Resize((image_size, image_size))
            original_images = transform(original_images)
            optimizer.zero_grad()

            # generate an image with a caption
            # generate the embeddings for the caption
            with torch.no_grad():
                input_ids = tokenizer(captions, return_tensors="pt", padding=True).input_ids.to(device=original_images.device)
                outputs = encoder.generate(input_ids, max_new_tokens=generator_embed_dim // 3).to(torch.float)

                # pad the outputs for the embedding dim/3
                embeddings = torch.zeros((batch_size, generator_embed_dim // 3))
                embeddings[:, :outputs.shape[1]] = outputs

            train_loop(discriminator, original_images, batch_size, discriminator_features, discriminator_loss_fn, accelerator,
                       optimizer, generator, ratios, embeddings, generator_loss_fn)
            
            # check if we need to do evaluations images
            step_count = step_count + 1
            if step_count % steps_per_eval == 0:
                eval(generator, tokenizer, encoder, eval_prompts, generator_embed_dim, eval_image_width, eval_image_height, original_images.device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--image_size', type=int, required=False, default=512)
    parser.add_argument('--discriminator_features', type=int, required=False, default=48)
    parser.add_argument('--generator_num_rows', type=int, required=False, default=64)
    parser.add_argument('--generator_num_heads', type=int, required=False, default=8)
    parser.add_argument('--generator_embed_dim', type=int, required=False, default=900)
    parser.add_argument('--generator_ratio_mult', type=int, required=False, default=100)
    parser.add_argument('--eval_image_height', type=int, required=False, default=512)
    parser.add_argument('--eval_image_width', type=int, required=False, default=512)
    parser.add_argument('--steps_per_eval', type=int, required=False, default=100)
    parser.add_argument('--eval_prompts', '-l', nargs='+', required=True)

    args = parser.parse_args()
    train(args)


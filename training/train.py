'''
'''

import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path
import sys
import gc
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
from tqdm.auto import tqdm, trange
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
    def __init__(self, batch_size : int, buckets : list[Bucket], embed_dim):
        self.batches = []
        self.embed_dim = embed_dim

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
        self.progress_bar = tqdm(self.batches)

    def __iter__(self):
        self.progress_bar = tqdm(self.batches)
        for batch in self.progress_bar:
            captions = []
            images = []
            for elem in batch:
                image_path, caption_path = elem
                with Image.open(image_path) as image:
                    caption = torch.load(caption_path, weights_only=True)
                    image = pil_to_tensor(image)

                    # put it between -1 and 1
                    image = image/256
                    captions.append(caption[0])
                    images.append(image)

            image_width = images[0].shape[2]
            image_height = images[0].shape[1]
            images_tensor = torch.zeros((len(batch), 3, image_height, image_width))
            embeddings = torch.zeros((len(captions), self.embed_dim))
            for i in range(len(batch)):
                images_tensor[i] = images[i]
                embeddings[i, :captions[i].shape[0]] = captions[i]
            yield (images_tensor, embeddings)

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
                
                # check if there is an associated .npz file
                filename = filepath.stem
                npz_file = filepath.parent.joinpath(filename + '.npz')
                if npz_file.exists():
                    # open the image to find its size
                    with Image.open(filepath) as image:
                        bucket = self.find_bucket(image.width, image.height)
                        if bucket == None:
                            self.buckets.append(Bucket(image.width, image.height))
                            bucket = self.buckets[-1]
                        
                        bucket.image_paths.append(filepath)
                        bucket.caption_paths.append(npz_file)

    def __getitem__(self, batch):
        # get the image size
        return batch

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

def train_loop(discriminator, original_images, batch_size, discriminator_features, discriminator_loss_fn,
               accelerator, optimizer, generator, ratios, embeddings, generator_loss_fn, single_step, sampler : BucketSampler):
    
    ############### discriminator ###########################################
    discriminator_total_loss = 0
    generator_total_loss = 0

    # push the real image into the discriminator
    freeze_model(generator)
    unfreeze_model(discriminator)
    disciminator_pred = discriminator(original_images, embeddings)
    real_preds = torch.ones((batch_size, discriminator_features)).to(device=disciminator_pred.device, dtype=disciminator_pred.dtype)
    loss = discriminator_loss_fn(disciminator_pred, real_preds)
    discriminator_total_loss = discriminator_total_loss + loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

    # push a fake image into the discriminator
    fake_images = torch.rand_like(original_images)
    disciminator_pred = discriminator(fake_images, embeddings)
    fake_preds = torch.zeros((batch_size, discriminator_features)).to(device=disciminator_pred.device, dtype=disciminator_pred.dtype)
    loss = discriminator_loss_fn(disciminator_pred, fake_preds)
    discriminator_total_loss = discriminator_total_loss + loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

    ############### generator ###########################################
    freeze_model(discriminator)
    unfreeze_model(generator)
    
    with torch.no_grad():
        z_fake_images = torch.fft.fftn(fake_images.to(torch.float), dim=(2, 3))

    if single_step:
        # only do for one column
        with torch.no_grad():
            start_column = torch.randint(0, generator.num_columns - 2, (1,))[0]
            z_images = generator(z_fake_images, ratios, embeddings, stop_column=start_column)
        z_images = generator(z_images, ratios, embeddings, start_column=start_column, stop_column=start_column+1)
        coeff = (start_column+1) / (generator.num_columns - 1)
        wanted_score = 2 * coeff / (1 + torch.pow(coeff, 2))
        preds = torch.full((batch_size, discriminator_features), wanted_score).to(device=disciminator_pred.device, dtype=disciminator_pred.dtype)
    else:
        z_images = generator(z_fake_images, ratios, embeddings)
        preds = real_preds

    # make sure we don't have complex images
    generated_images = torch.fft.ifftn(z_images, (generator.image_size, generator.image_size), dim=(2, 3))
    generated_images = generated_images.real
    generated_images = torch.nn.functional.relu(generated_images)
    disciminator_pred = discriminator(generated_images, embeddings)
    loss_d = discriminator_loss_fn(disciminator_pred, preds)
    loss_d = loss_d.to(dtype=generated_images.dtype)
    discriminator_total_loss = discriminator_total_loss + loss_d

    # rip the vram requirements for this
    #loss_g = generator_loss_fn(generated_images, original_images)
    #loss_g = loss_g.to(dtype=generated_images.dtype)
    loss_g = loss_d
    generator_total_loss = generator_total_loss + loss_g
    accelerator.backward(loss_g)
    #accelerator.backward(loss_d)
    optimizer.step()
    optimizer.zero_grad()

    # update the progress bar
    sampler.progress_bar.set_description(f'discriminator_loss={discriminator_total_loss}, generator_loss={generator_total_loss}', refresh=True)

def extract_features(tokenizer : T5Tokenizer, encoder : T5ForConditionalGeneration, dataset_folder, generator_embed_dim, device='cuda'):
    # we load all .txt images and put them on the disk
    for (dirpath, dirnames, filenames) in walk(dataset_folder):
        for filename in tqdm(filenames, desc='Extracting T5 embeddings'):
            filepath = pathlib.Path(dirpath).joinpath(filename)
            suffix = pathlib.Path(filepath).suffix

            if suffix != '.txt':
                continue
            
            # check if there is an associated .txt file
            caption = open(filepath).read()
            input_ids = tokenizer(caption, return_tensors="pt", padding=True).input_ids.to(device=device)
            outputs = encoder.generate(input_ids, max_new_tokens=generator_embed_dim // 3).to(torch.float)

            # save to disk
            stem = pathlib.Path(filepath).stem
            output = pathlib.Path(dirpath).joinpath(stem + '.npz')
            torch.save(outputs, output)


def eval(generator, prompts, generator_embed_dim, model_size, width, height, device='cuda'):
    images = []
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
    encoder = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small").to(device=device)

    with torch.no_grad():
        input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to(device=device)
        outputs = encoder.generate(input_ids, max_new_tokens=generator_embed_dim // 3).to(torch.float)

        # pad the outputs for the embedding dim/3
        embeddings = torch.zeros((outputs.shape[0], generator_embed_dim // 3))
        embeddings[:, :outputs.shape[1]] = outputs

        z_images = torch.rand((outputs.shape[0], 3, model_size, model_size), dtype=torch.complex64)
        z_images = torch.fft.fftn(z_images.to(torch.float), dim=(2, 3))
        ratios = torch.full((outputs.shape[0],), height / width)

        embeddings = embeddings.to(device=device)
        z_images = z_images.to(device=device)
        embeddings = embeddings.to(device=device)
        z_images = generator(z_images, ratios, embeddings)
        images = torch.fft.ifftn(z_images, (generator.image_size, generator.image_size), dim=(2, 3))
        images = images.real

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
        images = images * 255
        images = images.to(dtype=torch.uint8)

        i = 0
        for image in images:
            pil_image = to_pil_image(image)
            pil_image.save(f'test{i}.png')
            i = i + 1 
    
    del tokenizer
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
            
def train(args):
    dataset_folder = args.dataset_folder
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    steps_per_eval = args.steps_per_eval
    eval_prompts = args.eval_prompts
    single_step = args.single_step

    dataset = ImageDataset(dataset_folder)
    generator_embed_dim = args.generator_embed_dim
    bucket_sampler = BucketSampler(batch_size, dataset.buckets, generator_embed_dim // 3)

    # load the models
    image_size = args.image_size
    discriminator_features = args.discriminator_features
    generator_num_rows = args.generator_num_rows
    generator_num_heads = args.generator_num_heads
    generator_ratio_mult = args.generator_ratio_mult
    eval_image_height = args.eval_image_height
    eval_image_width = args.eval_image_width
    extract_t5 = args.extract_t5
    discriminator = Discriminator(image_size, discriminator_features, generator_embed_dim // 3, generator_num_heads)
    generator = Generator(image_size, generator_num_rows, generator_num_heads, generator_embed_dim, generator_ratio_mult)

    # setup the optimizers
    optimizer = AdamW(params=list(generator.parameters()) + list(discriminator.parameters()), lr=lr)
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
    encoder = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
    accelerator = Accelerator()
    tokenizer, encoder = accelerator.prepare(tokenizer, encoder)

    # first extract the features
    if extract_t5:
        extract_features(tokenizer, encoder, dataset_folder, generator_embed_dim)
    del encoder
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    def collate_fn(batch):
        return batch[0]
    
    dataloader = DataLoader(dataset, sampler=bucket_sampler, collate_fn=collate_fn)

    # prepare the models
    accelerator = Accelerator()
    
    dataloader, optimizer, discriminator, generator = \
        accelerator.prepare(dataloader, optimizer, discriminator, generator)

    # loss function for discriminator
    discriminator_loss_fn = torch.nn.BCELoss()

    # generated image loss for generator
    generator_loss_fn = torch.nn.MSELoss()

    # train loop
    step_count = 0
    for t in tqdm(range(epochs), desc='Epochs'):
        for batch in dataloader:
            original_images, captions = batch
            #original_images = original_images.to(dtype=torch.bfloat16)
            #captions = captions.to(dtype=torch.bfloat16)

            # calculate the ratios of the images
            image_width = original_images.shape[2]
            image_height = original_images.shape[1]
            batch_size = original_images.shape[0]
            ratios = torch.full((batch_size,), image_height / image_width)

            # rescale the original images down to a square image
            transform = Resize((image_size, image_size))
            original_images = transform(original_images)
            optimizer.zero_grad()

            embeddings = torch.zeros((batch_size, generator_embed_dim // 3))
            embeddings[:, :captions.shape[1]] = captions

            train_loop(discriminator, original_images, batch_size, discriminator_features, discriminator_loss_fn, accelerator,
                       optimizer, generator, ratios, embeddings, generator_loss_fn, single_step, bucket_sampler)
            
            # check if we need to do evaluations images
            step_count = step_count + 1
            if step_count % steps_per_eval == 0:
                eval(generator, eval_prompts, generator_embed_dim, image_size, eval_image_width, eval_image_height, original_images.device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--image_size', type=int, required=False, default=512)
    parser.add_argument('--discriminator_features', type=int, required=False, default=16)
    parser.add_argument('--generator_num_rows', type=int, required=False, default=128)
    parser.add_argument('--generator_num_heads', type=int, required=False, default=8)
    parser.add_argument('--generator_embed_dim', type=int, required=False, default=900)
    parser.add_argument('--generator_ratio_mult', type=int, required=False, default=100)
    parser.add_argument('--eval_image_height', type=int, required=False, default=512)
    parser.add_argument('--eval_image_width', type=int, required=False, default=512)
    parser.add_argument('--steps_per_eval', type=int, required=False, default=5)
    parser.add_argument('--extract_t5', action=argparse.BooleanOptionalAction)
    parser.add_argument('--single_step', action=argparse.BooleanOptionalAction)
    parser.add_argument('--eval_prompts', '-l', nargs='+', required=True)

    args = parser.parse_args()
    train(args)



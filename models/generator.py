from torch import nn
import torch
from accelerate import Accelerator
from torchvision.transforms import Resize

from pathlib import Path
import sys
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

class Transformer(nn.Module):
    def __init__(self, num_heads, embed_dim, image_height, column_width):
        # a sequence of calculations with attention layers
        # query is the image (tile_size x tile_size)
        # key is the embeds (2 * embed_size)
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=3 * column_width * image_height, num_heads=num_heads, kdim=embed_dim, batch_first=True)

        # layer norm
        self.layer_norm = nn.LayerNorm([3, image_height, column_width])

        # self attention
        self.self_attn = nn.MultiheadAttention(embed_dim=3 * column_width * image_height, num_heads=num_heads, batch_first=True)

        # relu activation layer
        self.relu = nn.ReLU()

        # tanh activation layer
        self.tanh = nn.Tanh()

    def forward(self, x, embeddings):
        # do some calculations with the embeddings
        query = torch.flatten(x, start_dim=1)
        key = embeddings.to(device=query.device, dtype=query.dtype)
        attn_output, attn_output_weights = self.cross_attn(query, key, query)

        # add input_freq and attention output
        x = x + attn_output.reshape(x.shape)
        x = self.layer_norm(x)
        #x = self.relu(x)

        # feedforward
        query = torch.flatten(x, start_dim=1)
        attn_output, attn_output_weights = self.self_attn(query, query, query)
        x = x + attn_output.reshape(x.shape)
        x = self.layer_norm(x)
        x = self.tanh(x)
        return x

class TileGenerator(nn.Module):
    def __init__(self, image_size, num_columns, num_heads, embed_dim, ratio_multiplier):
        super().__init__()

        # embedding lookup for the frequency component position
        self.image_width = image_size
        self.image_height = image_size
        self.num_columns = num_columns
        self.column_embed = nn.parameter.Parameter(torch.randn(num_columns, embed_dim // 3))
        self.column_width = self.image_width // num_columns

        # embedding lookup for the image ratios
        # this is in fixed point, ratio = h / w
        # i.e. 512x512 = 1000
        # 768x512 = 670
        # 512x768 = 1500
        # let's go up to 2 times the ratio_multiplier for the ratio
        self.ratio_embed = nn.parameter.Parameter(torch.randn(2 * ratio_multiplier, embed_dim // 3))
        self.ratio_multiplier = ratio_multiplier

        # transformer for both the real parts and imaginary parts
        self.real_transformer = Transformer(num_heads, embed_dim, self.image_height, self.column_width)
        self.imag_transformer = Transformer(num_heads, embed_dim, self.image_height, self.column_width)

    def forward(self, z_image, ratios, prompt_embeds, column):
        device = prompt_embeds.device
        dtype = prompt_embeds.dtype
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)

        # first compute the embedding for the position
        column_indices = torch.full((prompt_embeds.shape[0], ), column, device=device)
        column_embeds = self.column_embed[column_indices]
        column_embeds = column_embeds.to(dtype=dtype, device=device)

        # second compute the embedding for the ratio
        ratio_embeds = self.ratio_embed[ratios]
        ratio_embeds = ratio_embeds.to(dtype=dtype)

        # concat the 2 embeds together
        embeds = torch.cat((prompt_embeds.to(device=device, dtype=dtype), column_embeds, ratio_embeds.to(device=device, dtype=dtype)), dim=1)

        if column == self.num_columns - 1:
            max_x = torch.Tensor([self.image_height]).to(int)
        else:
            max_x = torch.Tensor([(column + 1) * self.column_width]).to(int)
        min_x = torch.Tensor([self.column_width * column]).to(int)
        input_freq = z_image[:, :, :, min_x:max_x]

        # do the calculus for real parts and imag parts
        real = input_freq.real
        imag = input_freq.imag

        real = self.real_transformer(real, embeds)
        imag = self.imag_transformer(imag, embeds)
        z = torch.complex(real, imag)

        z_image[:, :, :, min_x:max_x] = z_image[:, :, :, min_x:max_x] + z
        return z_image

class Generator(nn.Module):
    def __init__(self, image_size, num_rows, num_heads, embed_dim, ratio_multiplier):
        super().__init__()
        self.tile_generator = TileGenerator(image_size, num_rows, num_heads, embed_dim, ratio_multiplier)
        self.num_columns = num_rows
        self.ratio_multiplier = ratio_multiplier
        self.image_size = image_size

    def forward(self, z_images, ratios, prompt_embeds, start_column = 0, stop_column=-1):
        device = z_images.device
        dtype = z_images.dtype

        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        batch_size = ratios.shape[0]

        # first resize the images to square images
        transform = Resize((self.image_size, self.image_size))
        z_images = transform(z_images)

        # calculate the discrete real fft transform
        #z_images = torch.fft.fftn(images.to(dtype=torch.float32), s=(self.image_size, self.image_size))

        # calculate the ratio
        ratios_mult = ratios * self.ratio_multiplier
        ratios_mult = ratios.to(torch.int)

        if stop_column == -1:
            stop_column = self.num_columns

        for column in range(start_column, stop_column):
            z_images = self.tile_generator(z_images, ratios_mult, prompt_embeds, column)

        #images = torch.fft.ifftn(z_images, (self.image_size, self.image_size), dim=(2, 3))
        return z_images
    
if __name__ == '__main__':
    batch_size = 1

    image_size = 512
    num_rows = 64
    embed_dim = 900
    num_heads = 8
    ratio_multiplier = 100
    generator = Generator(image_size, num_rows, num_heads, embed_dim, ratio_multiplier)
    prompt_embeds = torch.zeros(batch_size, embed_dim // 3)

    image_width = 768
    image_height = 512
    ratios = torch.ones((batch_size))
    ratios[0] = image_height / image_width

    images = torch.rand((batch_size, 3, image_size, image_size))

    accelerator = Accelerator(mixed_precision='fp16')
    generator = accelerator.prepare(generator)

    images = images.to(dtype=torch.float16, device='cuda')
    with torch.no_grad():
        images = generator(images, ratios, prompt_embeds)

    # save the generator to see its size
    state_dict = generator.state_dict()
    new_state_dict = {}
    for key in state_dict.keys():
        new_state_dict[key] = state_dict[key].to(dtype=torch.float16)
    torch.save(new_state_dict, 'test.pth')



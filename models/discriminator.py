from torch import nn
import torch
from accelerate import Accelerator

class DetailDiscriminator(nn.Module):
    def __init__(self, image_size, conv_features, kernel_size, output_size, embed_dim, num_heads):
        super().__init__()

        # convolution layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, conv_features, padding=kernel_size//2, kernel_size=kernel_size+1, stride=1),
            nn.ReLU(),
            nn.Conv2d(conv_features, conv_features, padding=kernel_size//2, kernel_size=kernel_size+1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=image_size // output_size, stride=image_size // output_size, padding=0, dilation=1),
            nn.ReLU(),
        )

        # cross-attention with the prompt caption
        self.cross_attn = nn.MultiheadAttention(embed_dim=conv_features * output_size * output_size, num_heads=num_heads, kdim=embed_dim, batch_first=True)

        # dense layer
        self.forward_layer = nn.Sequential(
            nn.Linear(conv_features * output_size * output_size, conv_features * output_size * output_size),
            nn.ReLU(),
            nn.Linear(conv_features * output_size * output_size, conv_features),
            nn.Sigmoid()
        )

    def forward(self, image, embeddings):
        x = self.conv_layers(image)

        # cross-attention
        y = torch.flatten(x, start_dim=1)
        query = torch.flatten(x, start_dim=1)
        key = embeddings.to(device=query.device, dtype=query.dtype)
        attn_output, attn_output_weights = self.cross_attn(query, key, query)

        y = y + attn_output
        y = self.forward_layer(y)
        return y

class Discriminator(nn.Module):
    def __init__(self, image_size, conv_features, embed_dim, num_heads):
        super().__init__()
        self.low_detail_disc = DetailDiscriminator(image_size, conv_features, image_size // 4, conv_features, embed_dim, num_heads)
        self.medium_detail_disc = DetailDiscriminator(image_size, conv_features, image_size // 8, conv_features, embed_dim, num_heads)
        self.high_detail_disc = DetailDiscriminator(image_size, conv_features, image_size // 16, conv_features, embed_dim, num_heads)

    def forward(self, image, embeddings):
        low_details = self.low_detail_disc(image, embeddings)
        medium_details = self.medium_detail_disc(image, embeddings)
        high_detail_disc = self.high_detail_disc(image, embeddings)
        total = (low_details + medium_details + high_detail_disc) / 3
        total = torch.flatten(total, start_dim=1)
        return total

if __name__ == '__main__':
    batch_size = 1
    image_size = 512
    conv_features = 48
    discriminator = Discriminator(image_size, conv_features)

    image = torch.zeros((batch_size, 3, image_size, image_size))

    accelerator = Accelerator()
    discriminator = accelerator.prepare(discriminator)

    with torch.no_grad():
        image = image.to(accelerator.device)
        discriminator(image)

    # save the generator to see its size
    torch.save(discriminator.state_dict(), 'discriminator.pth')
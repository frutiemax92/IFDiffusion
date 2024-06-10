from torch import nn
import torch
from accelerate import Accelerator

class Discriminator(nn.Module):
    def __init__(self, image_size, conv_features):
        super().__init__()

        # convolution layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, conv_features, padding=1, dilation=1, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(conv_features, conv_features, padding=1, dilation=1, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(conv_features, conv_features, padding=1, dilation=1, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # layer norm
        self.layer_norm = nn.LayerNorm([conv_features, image_size, image_size])

        # dense layer
        self.forward_layer = nn.Sequential(
            nn.Linear(conv_features * image_size * image_size, conv_features),
            nn.Sigmoid()
        )

    def forward(self, image):
        device = image.device
        dtype = image.dtype
        x = self.conv_layers(image)
        x = self.layer_norm(x)
        x = self.forward_layer(torch.flatten(x, 1))
        x = x.to(device=device, dtype=dtype)
        return x

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
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.dense import DenseEncoder
from utils.text_utils import text_to_bits
from models.base import BasicEncoder
from models.residual import ResidualEncoder

if __name__ == "__main__":
    data_depth = 240  # this parameter sets the limit on the length of the text to be encoded
    hidden_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = DenseEncoder(data_depth, hidden_size, device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open('low_res_image.jpg')
    image_tensor = transform(image).unsqueeze(0)

    secret_text = "Hello, there!"
    data = text_to_bits(secret_text, data_depth)
    data_tensor = torch.FloatTensor(data).unsqueeze(0)

    output = encoder(image_tensor, data_tensor)

    output_image = output.squeeze(0).detach().cpu()
    output_image = (output_image * 0.5 + 0.5).clamp(0, 1)
    
    original_image = image_tensor.squeeze(0).cpu()
    original_image = (original_image * 0.5 + 0.5).clamp(0, 1)
    
    transforms.ToPILImage()(original_image).save('original_image.png')
    transforms.ToPILImage()(output_image).save('encoded_image.png')
    
    print(f"\nParameters:")
    print(f"- data_depth: {data_depth}")
    print(f"- hidden_size: {hidden_size}")
    print(f"- device: {device}")
    print(f"\nTensor shapes:")
    print(f"- image_tensor: {image_tensor.shape}")
    print(f"- data_tensor: {data_tensor.shape}") 
import torch
import torchvision
import numpy as np
import cv2 
from pathlib import Path
from PIL import Image
from src.models import UNet, PretrainedUNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

unet = PretrainedUNet(
    in_channels=1,
    out_channels=2, 
    batch_norm=True, 
    upscale_mode="bilinear"
)
models_folder = Path("models")

model_name = "unet-6v.pt"
unet.load_state_dict(torch.load(models_folder / model_name, map_location=torch.device("cpu")))
unet.to(device)
unet.eval();

device

origin_filename = "C:/Users/Chunayi/Desktop/001.png"

origin = Image.open(origin_filename).convert("P")
origin = torchvision.transforms.functional.resize(origin, (512, 512))
origin = torchvision.transforms.functional.to_tensor(origin) - 0.5

with torch.no_grad():
    origin = torch.stack([origin])
    origin = origin.to(device)
    out = unet(origin)
    softmax = torch.nn.functional.log_softmax(out, dim=1)
    out = torch.argmax(softmax, dim=1)
    
    origin = origin[0].to("cpu")
    out = out[0].to("cpu")

cv2.imwrite('output.png', np.array(out*255),[cv2.IMWRITE_PNG_COMPRESSION, 0])
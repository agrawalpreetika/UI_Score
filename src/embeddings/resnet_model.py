import torch
import torchvision.models as models
import torchvision.transforms as transforms
from utils.image_loader import load_image

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove FC layer
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def get_resnet_embedding(image_path):
    image = transform(load_image(image_path)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model(image)
    
    return embedding.cpu().numpy().flatten()
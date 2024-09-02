from typing import List, Tuple
import torch
import torchvision.transforms as T
from PIL import Image

def pred_class(model: torch.nn.Module, image: Image.Image, class_names: List[str], image_size: Tuple[int, int] = (224, 224)):
    # Create transformation for image
    image_transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(), 
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Transform and add an extra dimension to image
    transformed_image = image_transform(image).unsqueeze(dim=0)

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Make a prediction
        target_image_pred = model(transformed_image)
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Convert prediction probabilities to numpy array
    prob = target_image_pred_probs.cpu().numpy()

    return prob
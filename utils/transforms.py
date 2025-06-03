```python
import numpy as np
from PIL import Image
from torchvision import transforms

class CropToNonzeroBoundingBox:
    """
    Crop a PIL image to the minimal bounding box containing all non-background pixels.
    Assumes background is either all 0 or all 1 (can be changed via threshold).
    """
    def __init__(self, background_values=[0, 1]):
        self.background_values = background_values

    def __call__(self, img):
        # Convert to numpy array
        arr = np.array(img)
        # If grayscale, expand dims
        if arr.ndim == 2:
            arr = arr[None, ...]
        # If PIL image is RGB, shape is (H, W, 3)
        if arr.ndim == 3 and arr.shape[2] == 3:
            mask = ~np.all([(arr[..., c] == v) for v in self.background_values for c in range(3)], axis=0)
        else:
            # For single channel
            mask = ~np.isin(arr, self.background_values).all(axis=0)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return img  # No non-background, return as is
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1  # slices are exclusive at the top
        cropped = img.crop((x0, y0, x1, y1))
        return cropped

def get_transform(augment=True):
    transform = transforms.Compose([
        CropToNonzeroBoundingBox(background_values=[0, 1]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((1024, 768)),
        transforms.ToTensor()
    ]) if augment else transforms.Compose([
        CropToNonzeroBoundingBox(background_values=[0, 1]),
        transforms.Resize((1024, 768)),
        transforms.ToTensor()
    ])
    return transform
```
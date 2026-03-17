from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data_utils.config import DATASET_ROOT, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


class Places365FlatDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.image_paths = sorted(self.root.glob("*.jpg"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image_path.name, str(image_path)


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_places365_dataloader(batch_size=8):
    dataset = Places365FlatDataset(
        root=DATASET_ROOT,
        transform=get_transform()
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
import os
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms

# Transform to tensor (to check shape/validity)
to_tensor = transforms.ToTensor()

def check_image_dataset(dataset_dir):
    """
    Checks all images in a directory (recursively) to ensure they:
      - are loadable
      - can be converted to RGB
      - have exactly 3 channels
      - have finite values

    Args:
        dataset_dir (str): Path to the dataset directory.

    Returns:
        bad_images (list): List of tuples (file_path, reason) for problematic images.
    """
    VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    bad_images = []

    for root, _, files in os.walk(dataset_dir):
        for fname in files:
            if os.path.splitext(fname.lower())[1] in VALID_EXTS:
                fpath = os.path.join(root, fname)
                try:
                    img = Image.open(fpath).convert("RGB")
                    t = to_tensor(img)

                    if t.shape[0] != 3:
                        bad_images.append((fpath, f"Has {t.shape[0]} channels"))
                        continue

                    if not torch.isfinite(t).all():
                        bad_images.append((fpath, "Contains NaN or Inf"))
                        continue

                except (UnidentifiedImageError, OSError) as e:
                    bad_images.append((fpath, f"Unreadable: {e}"))

    if bad_images:
        print(f"\n Found {len(bad_images)} problematic images in {dataset_dir}:\n")
        for path, reason in bad_images:
            print(f"{path}  -->  {reason}")
    else:
        print(f"\n All images in {dataset_dir} are valid RGB (3-channel) and loadable.")

    return bad_images


# Example usage on multiple directories
if __name__ == "__main__":
    dirs = [
        "/home/du2/22CS30064/img2img-turbo/data/dataset_mmWave/test_A",
        "/home/du2/22CS30064/img2img-turbo/data/dataset_mmWave/test_B",
        "/home/du2/22CS30064/img2img-turbo/data/dataset_mmWave/train_A",
        "/home/du2/22CS30064/img2img-turbo/data/dataset_mmWave/train_B",
    ]
    for d in dirs:
        check_image_dataset(d)

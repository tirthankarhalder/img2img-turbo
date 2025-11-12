import os
from PIL import Image

def convert_rgba_to_rgb(input_dir, output_dir=None):
    """
    Convert only RGBA images in input_dir to RGB.
    If output_dir is None, images are overwritten in place.
    """
    if output_dir is None:
        output_dir = input_dir
    
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        save_path = os.path.join(output_dir, rel_path)
        os.makedirs(save_path, exist_ok=True)

        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                in_file = os.path.join(root, file)
                out_file = os.path.join(save_path, file)

                try:
                    with Image.open(in_file) as img:
                        if img.mode == "RGBA":
                            img = img.convert("RGB")
                            img.save(out_file)
                            print(f"Converted RGBA → RGB: {in_file}")
                        else:
                            img.save(out_file)
                            print(f"Skipped (already {img.mode}): {in_file}")
                except Exception as e:
                    print(f"Error processing {in_file}: {e}")

if __name__ == "__main__":
    # process both train_A and train_B
    print("It has started\n")
    convert_rgba_to_rgb("data/dataset_mmWave/train_A")
    print("A is done\n")
    convert_rgba_to_rgb("data/dataset_mmWave/train_B")
    print("B is done\n")

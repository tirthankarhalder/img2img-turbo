import os
from PIL import Image

def check_dataset_integrity(dataset_root):
    pairs = [('train_A', 'train_B'), ('test_A', 'test_B')]
    valid_exts = {'.jpg', '.jpeg', '.png'}

    for a_dir, b_dir in pairs:
        a_path = os.path.join(dataset_root, a_dir)
        b_path = os.path.join(dataset_root, b_dir)

        if not os.path.exists(a_path) or not os.path.exists(b_path):
            print(f"Missing {a_dir} or {b_dir}")
            continue

        # List image files only
        a_files = sorted([f for f in os.listdir(a_path)
                          if os.path.splitext(f)[1].lower() in valid_exts])
        b_files = sorted([f for f in os.listdir(b_path)
                          if os.path.splitext(f)[1].lower() in valid_exts])

        # Check count match
        if len(a_files) != len(b_files):
            print(f"⚠️ismatch in file count: {a_dir} has {len(a_files)}, {b_dir} has {len(b_files)}")
        else:
            print(f"{a_dir} and {b_dir} have the same number of images: {len(a_files)}")

        # Check filename match
        if a_files != b_files:
            print(f"⚠️Flenames do not match exactly between {a_dir} and {b_dir}")
        else:
            print(f"Filenames match between {a_dir} and {b_dir}")

        # Verify that all files are valid images and record their sizes
        for folder, files in [(a_dir, a_files), (b_dir, b_files)]:
            folder_path = os.path.join(dataset_root, folder)
            sizes = set()
            for fname in files:
                fpath = os.path.join(folder_path, fname)
                try:
                    with Image.open(fpath) as img:
                        img.verify()  # verify that file is readable
                    # reopen to get size (verify() closes the file)
                    with Image.open(fpath) as img:
                        sizes.add(img.size)  # (width, height)
                except Exception as e:
                    print(f"Corrupted or unreadable file: {fpath} ({e})")
            # Check if all sizes are the same
            if len(sizes) > 1:
                print(f"Images in {folder} have inconsistent sizes: {sizes}")
            else:
                print(f"All images in {folder} have consistent size: {list(sizes)[0]}")

    print("\nDataset integrity check complete.")
check_dataset_integrity("/home/du2/22CS30064/img2img-turbo/data/dataset_mmWave")

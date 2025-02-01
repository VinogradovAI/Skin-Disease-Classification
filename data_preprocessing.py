import pandas as pd
import os
import shutil
import albumentations as A
import cv2
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# Paths
DATA_PATH = Path("D:/ml_school/Skillfactory/CV/FP/raw_data/data")
PROCESSED_PATH = Path("D:/ml_school/Skillfactory/CV/FP/processed_data")
DATA_PARTS = ["train", "valid", "test"]

# Create output directories
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
for part in DATA_PARTS:
    for class_name in ["melanoma", "nevus", "seborrheic_keratosis"]:
        (PROCESSED_PATH / part / "img" / class_name).mkdir(parents=True, exist_ok=True)


# Load datasets
def load_data(data_part: str) -> pd.DataFrame:
    file_path = DATA_PATH / f"{data_part}_data.csv"
    return pd.read_csv(file_path)


# Define augmentations
AUGMENTATIONS = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=20, p=0.5),
    A.CLAHE(p=0.2),
    A.GaussianBlur(p=0.1)
])

# Additional augmentations for balancing classes
BALANCE_AUGMENTATIONS = A.Compose([
    A.HorizontalFlip(p=0.7),
    A.RandomBrightnessContrast(p=0.4),
    A.Rotate(limit=30, p=0.7),
    A.CLAHE(p=0.4),
    A.GaussianBlur(p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, p=0.4)
])

# Class distribution target (balancing to nevus sample count)
TARGET_SAMPLES = None


# Process images
def process_images(data_part: str):
    global TARGET_SAMPLES
    df = load_data(data_part)
    img_folder = DATA_PATH / data_part / "img"
    output_folder = PROCESSED_PATH / data_part / "img"

    class_counts = df['class'].value_counts().to_dict()
    if TARGET_SAMPLES is None:
        TARGET_SAMPLES = class_counts.get('nevus', max(class_counts.values()))
    generated_counts = Counter()

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {data_part}"):
        img_path = img_folder / row['image']
        class_label = row['class']
        class_folder = output_folder / class_label
        output_path = class_folder / row['image']

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if data_part == "train":
            # Apply standard augmentation
            augmented = AUGMENTATIONS(image=image)
            image = augmented["image"]
            cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # Generate additional samples for underrepresented classes
            if class_label in ["melanoma", "seborrheic_keratosis"]:
                for i in range(TARGET_SAMPLES // class_counts[class_label]):
                    aug_image = BALANCE_AUGMENTATIONS(image=image)
                    aug_output_path = class_folder / f"{row['image'].split('.')[0]}_aug_{i}.jpg"
                    cv2.imwrite(str(aug_output_path), cv2.cvtColor(aug_image["image"], cv2.COLOR_RGB2BGR))
                    generated_counts[class_label] += 1
        else:
            cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    if data_part == "train":
        print("Final class counts after augmentation:")
        final_counts = {cls: count + generated_counts[cls] for cls, count in class_counts.items()}
        for cls, count in final_counts.items():
            print(f"{cls}: {count}")


# Execute preprocessing
if __name__ == "__main__":
    for part in DATA_PARTS:
        process_images(part)
    print("Data preprocessing with class balancing completed!")
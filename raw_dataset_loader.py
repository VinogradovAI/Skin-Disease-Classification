import pandas as pd
import json
import zipfile
from pathlib import Path
from typing import Tuple, Optional

# Paths to data
ZIP_PATH = Path("C:/Users/vldmr/Downloads/CVresult/data.zip")
EXTRACT_TO = Path("D:/ml_school/Skillfactory/CV/FP/raw_data/")
DATA_PARTS = ["train", "test", "valid"]


# Extract zip archive
def extract_zip(zip_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Archive successfully extracted!")


# Extract metadata from JSON
def extract_tags_from_json(file_path: Path) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[str]]:
    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
            tags = data.get("tags", [])

            age = tags[0]['value'] if len(tags) > 0 else None
            sex = tags[1]['value'] if len(tags) > 1 else None
            primary_tag = tags[2]['name'] if len(tags) > 2 else None
            secondary_tag = tags[3]['name'] if len(tags) > 3 else None

            return age, sex, primary_tag, secondary_tag
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None, None


# Create dataset
def create_dataset(data_part: str) -> None:
    ann_dir = EXTRACT_TO / f"data/{data_part}/ann"
    img_dir = EXTRACT_TO / f"data/{data_part}/img"
    output_file = EXTRACT_TO / f"data/{data_part}_data.csv"

    data_records = []

    for file_path in ann_dir.glob("*.json"):
        age, sex, primary_tag, secondary_tag = extract_tags_from_json(file_path)
        img_name = file_path.stem

        if primary_tag == "melanoma":
            class_label = "melanoma"
        elif primary_tag == "nevus_or_seborrheic_keratosis" and secondary_tag == "melanoma_or_nevus":
            class_label = "nevus"
        elif primary_tag == "nevus_or_seborrheic_keratosis" and secondary_tag == "seborrheic_keratosis":
            class_label = "seborrheic_keratosis"
        else:
            class_label = "unknown"

        data_records.append([img_name, age, sex, class_label])

    df = pd.DataFrame(data_records, columns=["image", "age", "sex", "class"])
    df.to_csv(output_file, index=False)
    print(f"Dataset for {data_part} saved to {output_file}")


# Execution
if __name__ == "__main__":
    extract_zip(ZIP_PATH, EXTRACT_TO)
    for part in DATA_PARTS:
        create_dataset(part)

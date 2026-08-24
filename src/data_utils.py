"""
data_utils.py

Utility functions for loading the Driver Drowsiness Dataset (DDD),
extracting subject identifiers from filenames, performing a subject-level
train/val/test split (to avoid data leakage), and building image data
generators for training.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


# 1. Building the file index
def extract_subject_id(image_filename: str) -> str:
    """
    Extract the subject identifier from a DDD image filename.

    The DDD dataset encodes the subject as a leading letter (or letters)
    followed by a sequential frame number, e.g. 'A0001.png' -> subject 'A'.
    Matching is case-insensitive so that 'A' and 'a' across the Drowsy /
    Non Drowsy folders are treated as separate subjects, since DDD uses
    uppercase letters for the Drowsy class and lowercase letters for the
    Non Drowsy class by convention.

    Parameters
    ----------
    image_filename : str
        Image filename, e.g. 'A0001.png'.

    Returns
    -------
    str
        Subject identifier, e.g. 'A0001.png' -> 'A'.
    """
    filename_stem = Path(image_filename).stem
    subject_id_match = re.match(r"^[A-Za-z]+", filename_stem)
    if subject_id_match is None:
        raise ValueError(f"Could not extract subject id from filename: {image_filename}")
    return subject_id_match.group()


def build_file_index(raw_data_directory: str) -> pd.DataFrame:
    """
    Walk the raw DDD directory structure and build a DataFrame with one
    row per image, including its filepath, class label, and subject id.

    Expected directory structure:
        raw_data_directory/
            Drowsy/
                A0001.png
                A0002.png
                ...
            Non Drowsy/
                a0001.png
                a0002.png
                ...

    Parameters
    ----------
    raw_data_directory : str
        Path to the directory containing the 'Drowsy' and 'Non Drowsy'
        subfolders.

    Returns
    -------
    pd.DataFrame
        Columns: ['image_filepath', 'drowsiness_label', 'subject_id'].
        'drowsiness_label' is 1 for Drowsy, 0 for Non Drowsy.
    """
    raw_data_directory = Path(raw_data_directory)
    class_label_by_folder_name = {
        "Drowsy": 1,
        "Non Drowsy": 0,
    }

    image_records = []
    for folder_name, drowsiness_label in class_label_by_folder_name.items():
        class_folder_path = raw_data_directory / folder_name
        if not class_folder_path.exists():
            raise FileNotFoundError(f"Expected folder not found: {class_folder_path}")

        for image_path in sorted(class_folder_path.glob("*.png")):
            subject_id = extract_subject_id(image_path.name)
            image_records.append(
                {
                    "image_filepath": str(image_path),
                    "drowsiness_label": drowsiness_label,
                    "subject_id": f"{folder_name}_{subject_id}",
                }
            )

    drowsiness_dataset_df = pd.DataFrame.from_records(image_records)
    if drowsiness_dataset_df.empty:
        raise RuntimeError(
            f"No images found under {raw_data_directory}. Check the folder structure."
        )
    return drowsiness_dataset_df


# 2. Subject-level split
def subject_level_split(
    drowsiness_dataset_df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train / val / test sets at the SUBJECT level,
    not the image level. This prevents data leakage: images from the same
    subject (same person, same recording session) never appear in more
    than one split, which would otherwise let the model memorize
    subject-specific features instead of learning to generalize.

    In DDD, each subject belongs entirely to one class (drowsy-only or
    non-drowsy-only), and subjects vary widely in how many images they
    contribute (from ~109 to ~1,749). A plain subject-count stratified
    split can therefore still produce an image-level class imbalance
    across splits by chance. This function assigns subjects greedily,
    per class, to whichever split currently has the largest image-count
    deficit relative to its target ratio - balancing both subject count
    and image count per class across train/val/test.

    Parameters
    ----------
    drowsiness_dataset_df : pd.DataFrame
        Output of build_file_index(), must contain 'subject_id' and
        'drowsiness_label' columns.
    test_size : float
        Target fraction of IMAGES (per class) to hold out for testing.
    val_size : float
        Target fraction of IMAGES (per class) to hold out for validation.
    random_state : int
        Seed for reproducibility (used to shuffle subjects within each
        class before the size-based greedy assignment).

    Returns
    -------
    (train_subjects_df, val_subjects_df, test_subjects_df) : tuple of pd.DataFrame
        Image-level DataFrames, one row per image, filtered to the
        subjects assigned to each split.
    """
    train_size = 1.0 - test_size - val_size
    split_ratios = {"train": train_size, "val": val_size, "test": test_size}

    subject_summary_df = (
        drowsiness_dataset_df.groupby("subject_id")
        .agg(total_images=("drowsiness_label", "count"), drowsy_ratio=("drowsiness_label", "mean"))
        .reset_index()
    )
    subject_summary_df["subject_class"] = subject_summary_df["drowsy_ratio"].round().astype(int)

    rng = np.random.RandomState(random_state)
    assigned_subjects = {name: [] for name in split_ratios}

    for class_value in subject_summary_df["subject_class"].unique():
        class_subjects_df = subject_summary_df[
            subject_summary_df["subject_class"] == class_value
        ].sample(frac=1, random_state=rng)
        class_subjects_df = class_subjects_df.sort_values("total_images", ascending=False)

        class_total_images = class_subjects_df["total_images"].sum()
        target_images = {
            name: ratio * class_total_images for name, ratio in split_ratios.items()
        }
        # Tracked PER CLASS (reset for every class_value), not accumulated
        # across classes - otherwise the second class's deficits are computed
        # against images already assigned to the first class, which corrupts
        # the per-class balance this function is meant to achieve.
        assigned_images_this_class = {name: 0 for name in split_ratios}

        for _, subject_row in class_subjects_df.iterrows():
            deficits = {
                name: target_images[name] - assigned_images_this_class[name]
                for name in split_ratios
            }
            best_split = max(deficits, key=deficits.get)
            assigned_subjects[best_split].append(subject_row["subject_id"])
            assigned_images_this_class[best_split] += subject_row["total_images"]

    train_subjects_df = drowsiness_dataset_df[
        drowsiness_dataset_df["subject_id"].isin(assigned_subjects["train"])
    ].reset_index(drop=True)
    val_subjects_df = drowsiness_dataset_df[
        drowsiness_dataset_df["subject_id"].isin(assigned_subjects["val"])
    ].reset_index(drop=True)
    test_subjects_df = drowsiness_dataset_df[
        drowsiness_dataset_df["subject_id"].isin(assigned_subjects["test"])
    ].reset_index(drop=True)

    _assert_no_subject_overlap(train_subjects_df, val_subjects_df, test_subjects_df)

    return train_subjects_df, val_subjects_df, test_subjects_df


def _assert_no_subject_overlap(
    train_subjects_df: pd.DataFrame,
    val_subjects_df: pd.DataFrame,
    test_subjects_df: pd.DataFrame,
) -> None:
    """Sanity check: no subject_id should appear in more than one split."""
    train_subject_ids = set(train_subjects_df["subject_id"])
    val_subject_ids = set(val_subjects_df["subject_id"])
    test_subject_ids = set(test_subjects_df["subject_id"])

    assert train_subject_ids.isdisjoint(val_subject_ids), "Subject leakage between train and val!"
    assert train_subject_ids.isdisjoint(test_subject_ids), "Subject leakage between train and test!"
    assert val_subject_ids.isdisjoint(test_subject_ids), "Subject leakage between val and test!"


def print_split_summary(
    train_subjects_df: pd.DataFrame,
    val_subjects_df: pd.DataFrame,
    test_subjects_df: pd.DataFrame,
) -> None:
    """Print a quick summary of split sizes and class balance."""
    split_name_to_df = {
        "Train": train_subjects_df,
        "Val": val_subjects_df,
        "Test": test_subjects_df,
    }
    for split_name, split_subjects_df in split_name_to_df.items():
        image_count = len(split_subjects_df)
        subject_count = split_subjects_df["subject_id"].nunique()
        class_balance = (
            split_subjects_df["drowsiness_label"]
            .value_counts(normalize=True)
            .round(3)
            .to_dict()
        )
        print(
            f"{split_name:5s} | images: {image_count:6d} | subjects: {subject_count:3d} "
            f"| class balance (0=Non Drowsy, 1=Drowsy): {class_balance}"
        )


# 3. Image data generators
def build_data_generators(
    train_subjects_df: pd.DataFrame,
    val_subjects_df: pd.DataFrame,
    test_subjects_df: pd.DataFrame,
    image_size: tuple[int, int] = (160, 160),
    batch_size: int = 32,
):
    """
    Build Keras ImageDataGenerator flows for train, val, and test splits.

    Training data is augmented (rotation, zoom, brightness). Horizontal
    flip is disabled since face orientation is meaningful for this task.
    Validation and test data are only rescaled, never augmented.

    Parameters
    ----------
    train_subjects_df, val_subjects_df, test_subjects_df : pd.DataFrame
        Output of subject_level_split(). Must contain 'image_filepath'
        and 'drowsiness_label' columns.
    image_size : tuple[int, int]
        Target (height, width) for resizing images.
    batch_size : int
        Batch size for all generators.

    Returns
    -------
    (train_generator, val_generator, test_generator)
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # drowsiness_label column must be string for flow_from_dataframe with class_mode='binary'
    for split_subjects_df in (train_subjects_df, val_subjects_df, test_subjects_df):
        split_subjects_df["drowsiness_label"] = split_subjects_df["drowsiness_label"].astype(str)

    train_image_augmenter = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        zoom_range=0.1,
        brightness_range=(0.8, 1.2),
        width_shift_range=0.05,
        height_shift_range=0.05,
    )
    eval_image_rescaler = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_image_augmenter.flow_from_dataframe(
        train_subjects_df,
        x_col="image_filepath",
        y_col="drowsiness_label",
        target_size=image_size,
        color_mode="grayscale",
        class_mode="binary",
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )
    val_generator = eval_image_rescaler.flow_from_dataframe(
        val_subjects_df,
        x_col="image_filepath",
        y_col="drowsiness_label",
        target_size=image_size,
        color_mode="grayscale",
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False,
    )
    test_generator = eval_image_rescaler.flow_from_dataframe(
        test_subjects_df,
        x_col="image_filepath",
        y_col="drowsiness_label",
        target_size=image_size,
        color_mode="grayscale",
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False,
    )

    return train_generator, val_generator, test_generatorс
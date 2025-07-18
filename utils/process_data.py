import os
import nibabel as nib
import cv2
from tqdm import tqdm
from utils import normalize_image, save_image, resize_image
from config import EXTRACT_PATH, OUTPUT_PATH, SLICE_RANGE, SLICE_SIZE

def process_subject(subject_path, subject_id, label):
    """
    Processes the subject's MRI scans by extracting axial slices, normalizing, resizing,
    and saving them as PNG images.

    Args:
        subject_path (str): Path to the subject's MRI scans.
        subject_id (str): Subject identifier.
        label (int): The subject's CDR label (0 or 1).
    """
    t88_path = os.path.join(subject_path, 'MPRAGE', 'T88_111')

    img_path = None
    for f in os.listdir(t88_path):
        if f.endswith('masked_gfc.img') and 'fseg' not in f:
            img_path = os.path.join(t88_path, f)
            break

    if not img_path or not os.path.exists(img_path):
        print(f"No usable .img for {subject_id}")
        return

    img = nib.load(img_path).get_fdata()
    z_start = int(img.shape[2] * SLICE_RANGE[0])
    z_end = int(img.shape[2] * SLICE_RANGE[1])

    for i in range(z_start, z_end):
        slice_img = normalize_image(img[:, :, i])  # Axial slice
        resized = resize_image(slice_img, (SLICE_SIZE, SLICE_SIZE))
        out_path = os.path.join(OUTPUT_PATH, f'CDR_{label}', f'{subject_id}_z{i}.png')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_image(resized, out_path)

def process_all_subjects(extract_path):
    """
    Processes all subjects in the dataset by iterating over the directories and
    calling process_subject for each.

    Args:
        extract_path (str): Root path where all subjects' data is stored.
    """
    for disc in os.listdir(extract_path):
        disc_path = os.path.join(extract_path, disc)
        if not os.path.isdir(disc_path):
            continue

        for subj in tqdm(os.listdir(disc_path), desc=f'Processing {disc}'):
            subj_path = os.path.join(disc_path, subj)
            subj_id = subj

            # Add your metadata (cdr_map) for filtering
            if subj_id not in cdr_map:
                continue

            processed_path = os.path.join(subj_path, 'PROCESSED')

            if os.path.exists(processed_path):
                process_subject(processed_path, subj_id, cdr_map[subj_id])
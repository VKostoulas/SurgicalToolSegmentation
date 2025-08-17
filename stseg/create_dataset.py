import os
import ast
import io
import cv2
import zarr
import argparse
import zipfile
import tempfile
import numpy as np
from PIL import Image
from zarr.codecs import BloscCodec, BloscShuffle
from stseg.utils import clean_numpy_scalars


def main():
    parser = argparse.ArgumentParser(description="Create training dataset.")
    parser.add_argument("zip_dataset_path", type=str, help="Path to dataset zip.")
    parser.add_argument("save_dataset_path", type=str, help="Path of folder where dataset will be saved.")
    parser.add_argument("n_classes", type=int, help="Number of classes.")
    parser.add_argument("patch_size", type=ast.literal_eval, help="Patch size in format [W, H]")

    args = parser.parse_args()
    dataset_zip_path = args.zip_dataset_path
    dataset_save_path = args.save_dataset_path
    n_classes = args.n_classes
    patch_size = args.patch_size

    if not os.path.exists(os.path.join(dataset_save_path, 'data')):
        os.makedirs(os.path.join(dataset_save_path, 'data'))

    # create dataset with video frames and corresponding masks
    main_zip_folder = zipfile.ZipFile(dataset_zip_path)
    zip_file_names = [n for n in main_zip_folder.namelist() if n.lower().endswith(".zip")]
    for data_idx, zip_name in enumerate(zip_file_names):
        print(f"Processing {zip_name}...")
        with zipfile.ZipFile(io.BytesIO(main_zip_folder.read(zip_name))) as zf, tempfile.TemporaryDirectory() as td:
            mask_idx2name = {int(os.path.splitext(os.path.basename(n))[0]): n for n in zf.namelist() if n.endswith(".png")}

            vpath = os.path.join(td, "video_left.avi")
            with open(vpath, "wb") as f:
                f.write(zf.read("video_left.avi"))

            cap = cv2.VideoCapture(vpath)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"    {zip_name}: {w}x{h}, fps={fps}, frames={nfr}, mask frames={len(mask_idx2name)} (1/60th of frames)")

            file_save_path = os.path.join(dataset_save_path, 'data', f"data_{data_idx + 1:03d}.zarr")
            compressor = BloscCodec(cname='zstd', clevel=3, shuffle=BloscShuffle.bitshuffle)
            image_chunks = (3, 1, *patch_size)
            mask_chunks = (1, *patch_size)
            z_file = zarr.open(file_save_path, mode='w')
            n = len(mask_idx2name)
            img_ar = z_file.create_array(name='image', shape=(3, n, h, w), chunks=image_chunks, dtype=np.float32,
                                       compressors=compressor, overwrite=True)
            mask_ar = z_file.create_array(name='mask', shape=(n, h, w), chunks=mask_chunks, dtype=np.uint8,
                                        compressors=compressor, overwrite=True)

            class_locations = {int(lbl): [] for lbl in range(1, n_classes + 1)}
            for i, idx in enumerate(sorted(mask_idx2name)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                _, frame = cap.read()
                mask = np.array(Image.open(io.BytesIO(zf.read(mask_idx2name[idx])))).max(-1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32)
                frame /= 255.0
                # # optional: normalize with imageNet stats
                # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
                # std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
                # frame = (frame - mean) / std
                img_ar[:, i] = frame
                mask_ar[i] = mask

                for lbl in range(1, n_classes + 1):
                    slice_mask = mask == lbl
                    slice_coords = np.argwhere(slice_mask)

                    if slice_coords.shape[0] == 0:
                        continue  # no voxels for this label in this slice

                    if slice_coords.shape[0] > 50:
                        indices = np.random.choice(slice_coords.shape[0], 50, replace=False)
                        sampled = slice_coords[indices]
                    else:
                        sampled = slice_coords

                    # add Z as the first coordinate
                    sampled = [(i, y, x) for y, x in sampled]
                    class_locations[int(lbl)].extend(sampled)

            properties = {'class_locations': class_locations}
            z_file.attrs['properties'] = clean_numpy_scalars(properties)

            print(f"    Saved image and mask at {file_save_path}")
            cap.release()

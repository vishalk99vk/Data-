import streamlit as st
import numpy as np
import os
import zipfile
from scipy.ndimage import gaussian_filter
from io import BytesIO
from PIL import Image


def apply_adjustments_no_hue(img, brightness_map, contrast_map):
    b, g, r = cv2.split(img.astype(np.float32))
    b = contrast_map * b + (brightness_map - 1) * 100
    g = contrast_map * g + (brightness_map - 1) * 100
    r = contrast_map * r + (brightness_map - 1) * 100
    return cv2.merge([np.clip(b, 0, 255), np.clip(g, 0, 255), np.clip(r, 0, 255)]).astype(np.uint8)

def apply_tamper_shadow(img, intensity=0.5, radius_fraction=0.5):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    radius = int(min(h, w) * radius_fraction)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = np.clip((radius - dist) / radius, 0, 1) ** 2
    shadow = 1 - intensity * mask[..., np.newaxis]
    return (img.astype(np.float32) * shadow).astype(np.uint8)

def match_image_gridwise_smooth_no_hue(ref_img, client_img, grid=(6,6), sigma=10):
    h, w = ref_img.shape[:2]
    gh, gw = h // grid[0], w // grid[1]
    brightness_map = np.zeros((grid[0], grid[1]), dtype=np.float32)
    contrast_map = np.zeros((grid[0], grid[1]), dtype=np.float32)

    for i in range(grid[0]):
        for j in range(grid[1]):
            y1, y2 = i * gh, (i + 1) * gh
            x1, x2 = j * gw, (j + 1) * gw
            ref_patch = ref_img[y1:y2, x1:x2]
            cli_patch = client_img[y1:y2, x1:x2]
            ref_gray = cv2.cvtColor(ref_patch, cv2.COLOR_BGR2GRAY)
            cli_gray = cv2.cvtColor(cli_patch, cv2.COLOR_BGR2GRAY)
            ref_mean, ref_std = np.mean(ref_gray), np.std(ref_gray)
            cli_mean, cli_std = np.mean(cli_gray), np.std(cli_gray)
            brightness_map[i, j] = (ref_mean - cli_mean) / 100 + 1
            contrast_map[i, j] = ref_std / (cli_std + 1e-5)

    brightness_map_up = cv2.resize(brightness_map, (w, h), interpolation=cv2.INTER_LINEAR)
    contrast_map_up = cv2.resize(contrast_map, (w, h), interpolation=cv2.INTER_LINEAR)
    brightness_map_smooth = gaussian_filter(brightness_map_up, sigma=sigma)
    contrast_map_smooth = gaussian_filter(contrast_map_up, sigma=sigma)
    adjusted = apply_adjustments_no_hue(client_img, brightness_map_smooth, contrast_map_smooth)
    return apply_tamper_shadow(adjusted, intensity=0.4, radius_fraction=0.5)

def convert_to_image(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

st.title("Batch Image Color Matching Tool (No Hue + Shadow Effect)")

ref_imgs = st.file_uploader("Upload Reference Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
cli_imgs = st.file_uploader("Upload Client Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if st.button("Process Images") and ref_imgs and cli_imgs:
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zipf:
        for r_idx, r_file in enumerate(ref_imgs):
            ref_np = np.array(Image.open(r_file).convert("RGB"))
            ref_cv = cv2.cvtColor(ref_np, cv2.COLOR_RGB2BGR)

            for c_idx, c_file in enumerate(cli_imgs):
                cli_np = np.array(Image.open(c_file).convert("RGB"))
                cli_cv = cv2.cvtColor(cli_np, cv2.COLOR_RGB2BGR)
                cli_resized = cv2.resize(cli_cv, (ref_cv.shape[1], ref_cv.shape[0]))

                output = match_image_gridwise_smooth_no_hue(ref_cv, cli_resized)

                result_img = convert_to_image(output)
                img_name = f"{r_file.name.split('.')[0]}__{c_file.name.split('.')[0]}.jpg"
                buffer = BytesIO()
                result_img.save(buffer, format="JPEG")
                zipf.writestr(img_name, buffer.getvalue())

    st.success("Processing Complete!")
    st.download_button("Download All as ZIP", data=zip_buffer.getvalue(), file_name="processed_images.zip")


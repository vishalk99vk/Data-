import streamlit as st
import numpy as np
import zipfile
import cv2
from scipy.ndimage import gaussian_filter
from io import BytesIO
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor

# 1. Resize utility to downscale large images
def resize_if_large(img, max_dim=1024):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

# 2. Efficient image loading using cache
@st.cache_data
def load_image(file):
    return np.array(Image.open(file).convert("RGB"))

# 3. Adjust brightness and contrast without hue manipulation
def apply_adjustments_no_hue(img, brightness_map, contrast_map):
    b, g, r = cv2.split(img.astype(np.float32))
    b = contrast_map * b + (brightness_map - 1) * 100
    g = contrast_map * g + (brightness_map - 1) * 100
    r = contrast_map * r + (brightness_map - 1) * 100
    return cv2.merge([np.clip(b, 0, 255), np.clip(g, 0, 255), np.clip(r, 0, 255)]).astype(np.uint8)

# 4. Apply circular shadow to the image
def apply_tamper_shadow(img, intensity=0.5, radius_fraction=0.5):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    radius = int(min(h, w) * radius_fraction)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = np.clip((radius - dist) / radius, 0, 1) ** 2
    shadow = 1 - intensity * mask[..., np.newaxis]
    return (img.astype(np.float32) * shadow).astype(np.uint8)

# 5. Main matching function with grid-based brightness & contrast mapping
def match_image_gridwise_smooth_no_hue(ref_img, client_img, grid=(6,6), sigma=5):
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

    # Fixed sigma smoothing
    brightness_map_smooth = gaussian_filter(brightness_map_up, sigma=5)
    contrast_map_smooth = gaussian_filter(contrast_map_up, sigma=5)

    adjusted = apply_adjustments_no_hue(client_img, brightness_map_smooth, contrast_map_smooth)
    return apply_tamper_shadow(adjusted, intensity=0.4, radius_fraction=0.5)

# Function to process one pair of images for parallel processing
def process_pair(ref_np, cli_np):
    ref_cv = resize_if_large(cv2.cvtColor(ref_np, cv2.COLOR_RGB2BGR))
    cli_cv = resize_if_large(cv2.cvtColor(cli_np, cv2.COLOR_RGB2BGR))
    cli_resized = cv2.resize(cli_cv, (ref_cv.shape[1], ref_cv.shape[0]))
    return match_image_gridwise_smooth_no_hue(ref_cv, cli_resized)

# ---------------- Streamlit UI ----------------
st.title("📸 Batch Image Color Matching Tool (No Hue + Shadow)")

ref_imgs = st.file_uploader("📂 Upload Reference Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
cli_imgs = st.file_uploader("📂 Upload Client Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Preview one reference and one client image
if ref_imgs and cli_imgs:
    try:
        st.image(load_image(ref_imgs[0]), caption="Sample Reference Image", width=300)
        st.image(load_image(cli_imgs[0]), caption="Sample Client Image", width=300)
    except Exception as e:
        st.warning(f"Preview failed: {e}")

if st.button("🚀 Process Images") and ref_imgs and cli_imgs:
    zip_buffer = BytesIO()
    total = len(ref_imgs) * len(cli_imgs)
    count = 0
    progress_bar = st.progress(0)

    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zipf:
        with ThreadPoolExecutor() as executor:
            futures = []
            for r_file in ref_imgs:
                try:
                    ref_np = load_image(r_file)
                except Exception as e:
                    st.error(f"Error loading reference image {r_file.name}: {e}")
                    continue

                for c_file in cli_imgs:
                    try:
                        cli_np = load_image(c_file)
                    except Exception as e:
                        st.error(f"Error loading client image {c_file.name}: {e}")
                        continue
                    futures.append((r_file.name, c_file.name, executor.submit(process_pair, ref_np, cli_np)))

            for r_name, c_name, future in futures:
                try:
                    output = future.result()
                except Exception as e:
                    st.error(f"Error processing {r_name} and {c_name}: {e}")
                    continue

                # Clean file names
                base_r = os.path.splitext(r_name)[0]
                base_c = os.path.splitext(c_name)[0]
                img_name = f"{base_r}__{base_c}.jpg"

                # Save image to zip
                _, img_encoded = cv2.imencode('.jpg', output)
                zipf.writestr(img_name, img_encoded.tobytes())

                count += 1
                progress_bar.progress(count / total)

                # Free memory
                del output

    st.success("✅ Processing Complete!")
    st.download_button("📦 Download All as ZIP", data=zip_buffer.getvalue(), file_name="processed_images.zip")

# Custom button styling (optional)
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #0099ff;
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

import streamlit as st
import numpy as np
import zipfile
import cv2
from scipy.ndimage import gaussian_filter
from io import BytesIO
from PIL import Image

# Resize utility
def resize_if_large(img, max_dim=1024):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

# Load image from file-like
@st.cache_data
def load_image(file):
    return np.array(Image.open(file).convert("RGB"))

# Adjust brightness and contrast (no hue)
def apply_adjustments_no_hue(img, brightness_map, contrast_map):
    b, g, r = cv2.split(img.astype(np.float32))
    b = contrast_map * b + (brightness_map - 1) * 100
    g = contrast_map * g + (brightness_map - 1) * 100
    r = contrast_map * r + (brightness_map - 1) * 100
    return cv2.merge([np.clip(b, 0, 255), np.clip(g, 0, 255), np.clip(r, 0, 255)]).astype(np.uint8)

# Circular shadow
def apply_tamper_shadow(img, intensity=0.5, radius_fraction=0.5):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    radius = int(min(h, w) * radius_fraction)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = np.clip((radius - dist) / radius, 0, 1) ** 2
    shadow = 1 - intensity * mask[..., np.newaxis]
    return (img.astype(np.float32) * shadow).astype(np.uint8)

# Main matching function
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
    brightness_map_smooth = gaussian_filter(brightness_map_up, sigma=sigma)
    contrast_map_smooth = gaussian_filter(contrast_map_up, sigma=sigma)
    adjusted = apply_adjustments_no_hue(client_img, brightness_map_smooth, contrast_map_smooth)
    return apply_tamper_shadow(adjusted, intensity=0.4, radius_fraction=0.5)

# Effects
def apply_tint(img, tint_color=(0, 100, 150), alpha=0.3):
    overlay = np.full(img.shape, tint_color, dtype=np.uint8)
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

def apply_3d_rotation(img, angle=30):
    h, w = img.shape[:2]
    offset = int(w * 0.2 * (angle / 60))
    pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    pts2 = np.float32([[offset,0], [w - offset,0], [0,h], [w,h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h))

def apply_gaussian_blur(img):
    return cv2.GaussianBlur(img, (15, 15), 0)

def apply_reflection(img):
    return cv2.flip(img, 1)

def apply_human_reflection(img):
    reflected = apply_reflection(img)
    return cv2.convertScaleAbs(reflected, alpha=1.1, beta=20)

def apply_default_effect(img, effect_name, rotation_angle=30, tint_alpha=0.3):
    if effect_name == "Gaussian Blur":
        return apply_gaussian_blur(img)
    elif effect_name == "Reflection":
        return apply_reflection(img)
    elif effect_name == "Human Reflection":
        return apply_human_reflection(img)
    elif effect_name == "Tint":
        return apply_tint(img, alpha=tint_alpha)
    elif effect_name == "3D Rotation":
        return apply_3d_rotation(img, angle=rotation_angle)
    else:
        return img

# Extract images from ZIP
def extract_images_from_zip(zip_file):
    images = []
    with zipfile.ZipFile(zip_file) as z:
        for file in z.namelist():
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                with z.open(file) as f:
                    img = np.array(Image.open(f).convert("RGB"))
                    images.append((file, img))
    return images

# --- Streamlit UI ---
st.title("📸 Bulk Image Color Matching with ZIP for Reference & Client Images")

# Upload ref images or ZIP
ref_upload_type = st.radio("Reference images upload type:", ("Individual Images", "ZIP File"))

ref_imgs = []
if ref_upload_type == "Individual Images":
    ref_files = st.file_uploader("Upload Reference Images", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if ref_files:
        for f in ref_files:
            img_np = load_image(f)
            img_cv = resize_if_large(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            ref_imgs.append((f.name, img_cv))
else:
    ref_zip = st.file_uploader("Upload Reference Images ZIP", type=["zip"])
    if ref_zip:
        ref_imgs_raw = extract_images_from_zip(ref_zip)
        for name, img in ref_imgs_raw:
            img_cv = resize_if_large(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            ref_imgs.append((name, img_cv))

# Upload client images or ZIP
client_upload_type = st.radio("Client images upload type:", ("Individual Images", "ZIP File"))

client_imgs = []
if client_upload_type == "Individual Images":
    client_files = st.file_uploader("Upload Client Images", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if client_files:
        for f in client_files:
            img_np = load_image(f)
            img_cv = resize_if_large(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            client_imgs.append((f.name, img_cv))
else:
    client_zip = st.file_uploader("Upload Client Images ZIP", type=["zip"])
    if client_zip:
        client_imgs_raw = extract_images_from_zip(client_zip)
        for name, img in client_imgs_raw:
            img_cv = resize_if_large(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            client_imgs.append((name, img_cv))

# Default effect selection if no references
default_effects = ["Gaussian Blur", "Reflection", "Human Reflection", "Tint", "3D Rotation"]
selected_effect = None
rotation_angle = 30
tint_alpha = 0.3

if not ref_imgs:
    selected_effect = st.selectbox("No reference images uploaded. Select default effect:", default_effects)
    if selected_effect == "3D Rotation":
        rotation_angle = st.slider("Select Rotation Angle (degrees):", min_value=0, max_value=60, value=30)
    elif selected_effect == "Tint":
        tint_alpha = st.slider("Select Tint Intensity (alpha):", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

if st.button("🚀 Process Images"):

    if not client_imgs:
        st.error("Please upload client images to process.")
    else:
        zip_buffer = BytesIO()
        total_count = max(1, len(ref_imgs)) * len(client_imgs)
        progress = 0
        progress_bar = st.progress(0)

        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zipf:
            if ref_imgs:
                # Process every ref with every client
                for r_name, r_img in ref_imgs:
                    for c_name, c_img in client_imgs:
                        # Resize client to ref size for consistency
                        c_resized = cv2.resize(c_img, (r_img.shape[1], r_img.shape[0]))
                        output = match_image_gridwise_smooth_no_hue(r_img, c_resized)
                        out_name = f"{r_name.split('.')[0]}__{c_name.split('.')[0]}.jpg"
                        _, encoded_img = cv2.imencode(".jpg", output)
                        zipf.writestr(out_name, encoded_img.tobytes())
                        progress += 1
                        progress_bar.progress(progress / total_count)
            else:
                # No references, apply default effect to all client images
                for c_name, c_img in client_imgs:
                    output = apply_default_effect(c_img, selected_effect, rotation_angle, tint_alpha)
                    out_name = f"defaultEffect__{c_name.split('.')[0]}.jpg"
                    _, encoded_img = cv2.imencode(".jpg", output)
                    zipf.writestr(out_name, encoded_img.tobytes())
                    progress += 1
                    progress_bar.progress(progress / total_count)

        st.success("✅ Processing complete!")
        st.download_button("📦 Download all processed images as ZIP", data=zip_buffer.getvalue(), file_name="processed_images.zip")

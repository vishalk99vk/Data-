import streamlit as st
import numpy as np
import zipfile
import cv2
from io import BytesIO
from PIL import Image

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
def match_image_gridwise_smooth_no_hue(ref_img, client_img, grid=(6,6)):
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
    adjusted = apply_adjustments_no_hue(client_img, brightness_map_up, contrast_map_up)
    return apply_tamper_shadow(adjusted, intensity=0.4, radius_fraction=0.5)

# --- Default effects functions ---

def apply_gaussian_blur(img, ksize=7):
    if ksize % 2 == 0:  # kernel size must be odd
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def apply_reflection(img):
    # Simple horizontal flip as reflection example
    return cv2.flip(img, 1)

def apply_human_reflection(img):
    # Simulate human reflection by blending image with blurred flipped version
    flipped = cv2.flip(img, 1)
    blurred = cv2.GaussianBlur(flipped, (21, 21), 10)
    return cv2.addWeighted(img, 0.7, blurred, 0.3, 0)

def apply_tint(img, alpha=0.3, color=(0, 0, 255)):  # default blue tint
    overlay = np.full_like(img, color, dtype=np.uint8)
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

def apply_3d_rotation(img, angle=30):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated

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

# ---------------- Streamlit UI ----------------
st.title("📸 Batch Image Color Matching Tool (No Hue + Shadow)")

ref_imgs_files = st.file_uploader("📂 Upload Reference Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
cli_imgs_files = st.file_uploader("📂 Upload Client Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Load images to arrays for easier processing
ref_imgs = []
if ref_imgs_files:
    for f in ref_imgs_files:
        img = load_image(f)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = resize_if_large(img)
        ref_imgs.append((f.name, img))

cli_imgs = []
if cli_imgs_files:
    for f in cli_imgs_files:
        img = load_image(f)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = resize_if_large(img)
        cli_imgs.append((f.name, img))

# Show default effect options only if no reference images
default_effects = ["Gaussian Blur", "Reflection", "Human Reflection", "Tint", "3D Rotation"]
selected_effect = None
rotation_angle = 30
tint_alpha = 0.3

if not ref_imgs and cli_imgs:
    selected_effect = st.selectbox("No reference images uploaded. Select default effect:", default_effects)
    if selected_effect == "3D Rotation":
        rotation_angle = st.slider("Select Rotation Angle (degrees)", 0, 60, 30)
    elif selected_effect == "Tint":
        tint_alpha = st.slider("Select Tint Intensity (alpha)", 0.0, 1.0, 0.3, 0.05)

if st.button("🚀 Process Images"):
    if not cli_imgs:
        st.error("Please upload client images to process.")
    else:
        zip_buffer = BytesIO()
        total_count = max(1, len(ref_imgs)) * len(cli_imgs)
        progress = 0
        progress_bar = st.progress(0)

        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zipf:
            if ref_imgs:
                for r_name, r_img in ref_imgs:
                    for c_name, c_img in cli_imgs:
                        c_resized = cv2.resize(c_img, (r_img.shape[1], r_img.shape[0]))
                        output = match_image_gridwise_smooth_no_hue(r_img, c_resized)
                        out_name = f"{r_name.split('.')[0]}__{c_name.split('.')[0]}.jpg"
                        _, encoded_img = cv2.imencode(".jpg", output)
                        zipf.writestr(out_name, encoded_img.tobytes())
                        progress += 1
                        progress_bar.progress(progress / total_count)
            else:
                # No reference images, apply default effect on all client images
                for c_name, c_img in cli_imgs:
                    output = apply_default_effect(c_img, selected_effect, rotation_angle, tint_alpha)
                    out_name = f"defaultEffect__{c_name.split('.')[0]}.jpg"
                    _, encoded_img = cv2.imencode(".jpg", output)
                    zipf.writestr(out_name, encoded_img.tobytes())
                    progress += 1
                    progress_bar.progress(progress / total_count)

        st.success("✅ Processing complete!")
        st.download_button("📦 Download all processed images as ZIP", data=zip_buffer.getvalue(), file_name="processed_images.zip")

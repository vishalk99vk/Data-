import streamlit as st
import numpy as np
import zipfile
import cv2
from scipy.ndimage import gaussian_filter
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

# --------- Additional Effects ---------

def apply_tint(img, tint_color=(0, 100, 150), alpha=0.3):
    overlay = np.full(img.shape, tint_color, dtype=np.uint8)
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

def apply_3d_rotation(img, angle=30):
    h, w = img.shape[:2]
    offset = int(w * 0.2 * (angle / 60))  # scale offset with angle
    pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    pts2 = np.float32([[offset,0], [w - offset,0], [0,h], [w,h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h))

# Dummy placeholders for reflection effects (you can implement your own)
def apply_gaussian_blur(img):
    return cv2.GaussianBlur(img, (15, 15), 0)

def apply_reflection(img):
    # Simple horizontal flip as dummy reflection effect
    return cv2.flip(img, 1)

def apply_human_reflection(img):
    # For demo, apply reflection + slight brightness increase
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

# ---------------- Streamlit UI ----------------
st.title("📸 Batch Image Color Matching Tool (No Hue + Shadow)")

ref_imgs = st.file_uploader("📂 Upload Reference Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
cli_imgs = st.file_uploader("📂 Upload Client Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Only show sigma internally (no slider per your request, keep default)
sigma = 5

# Default effect if no reference images provided
default_effects = ["Gaussian Blur", "Reflection", "Human Reflection", "Tint", "3D Rotation"]
selected_effect = None
rotation_angle = 30
tint_alpha = 0.3

if not ref_imgs:
    selected_effect = st.selectbox("Select Default Effect (No Reference Images Uploaded):", default_effects)
    if selected_effect == "3D Rotation":
        rotation_angle = st.slider("Select Rotation Angle (degrees):", min_value=0, max_value=60, value=30)
    elif selected_effect == "Tint":
        tint_alpha = st.slider("Select Tint Intensity (alpha):", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

if st.button("🚀 Process Images") and cli_imgs:
    zip_buffer = BytesIO()
    total = 0
    count = 0
    progress_bar = st.progress(0)

    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zipf:

        # If reference images exist, run matching
        if ref_imgs:
            total = len(ref_imgs) * len(cli_imgs)
            for r_file in ref_imgs:
                ref_np = load_image(r_file)
                ref_cv = resize_if_large(cv2.cvtColor(ref_np, cv2.COLOR_RGB2BGR))

                for c_file in cli_imgs:
                    cli_np = load_image(c_file)
                    cli_cv = resize_if_large(cv2.cvtColor(cli_np, cv2.COLOR_RGB2BGR))
                    cli_resized = cv2.resize(cli_cv, (ref_cv.shape[1], ref_cv.shape[0]))

                    output = match_image_gridwise_smooth_no_hue(ref_cv, cli_resized, sigma=sigma)

                    _, img_encoded = cv2.imencode('.jpg', output)
                    img_name = f"{r_file.name.split('.')[0]}__{c_file.name.split('.')[0]}.jpg"
                    zipf.writestr(img_name, img_encoded.tobytes())

                    count += 1
                    progress_bar.progress(count / total)

        # If no reference images, apply default effect only on client images
        else:
            total = len(cli_imgs)
            for c_file in cli_imgs:
                cli_np = load_image(c_file)
                cli_cv = resize_if_large(cv2.cvtColor(cli_np, cv2.COLOR_RGB2BGR))

                output = apply_default_effect(cli_cv, selected_effect, rotation_angle, tint_alpha)

                _, img_encoded = cv2.imencode('.jpg', output)
                img_name = f"defaultEffect__{c_file.name.split('.')[0]}.jpg"
                zipf.writestr(img_name, img_encoded.tobytes())

                count += 1
                progress_bar.progress(count / total)

    st.success("✅ Processing Complete!")
    st.download_button("📦 Download All as ZIP", data=zip_buffer.getvalue(), file_name="processed_images.zip")

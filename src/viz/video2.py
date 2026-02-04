# # import os, re
# # from pathlib import Path
# # import numpy as np
# # import imageio
# # from PIL import Image

# # FRAMES_DIR = Path("storm_frames")      # folder with your PNGs
# # OUT_MP4 = Path("storm_worldmap_30s.mp4")
# # DURATION_SEC = 30.0

# # # Optional: pan/zoom crop (set to False if you want full frame)
# # DO_PAN = True

# # # Pan settings: define a crop window size (in pixels) and how it moves
# # # If your frames are e.g. 1280x720, a crop like 960x540 looks like zoom-in.
# # CROP_W, CROP_H = 960, 540
# # PAN_START = (0.10, 0.55)   # (x_frac, y_frac) top-left start as fraction of frame size
# # PAN_END   = (0.35, 0.35)   # (x_frac, y_frac) top-left end   as fraction of frame size

# # def natural_sort_key(name: str):
# #     nums = re.findall(r"\d+", name)
# #     return int(nums[-1]) if nums else 0

# # def list_frames(folder: Path):
# #     files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
# #     files.sort(key=natural_sort_key)
# #     return [folder / f for f in files]

# # def crop_pan(img: Image.Image, t: float):
# #     # t in [0,1]
# #     w, h = img.size
# #     cw, ch = min(CROP_W, w), min(CROP_H, h)

# #     sx, sy = PAN_START
# #     ex, ey = PAN_END

# #     x0 = int((sx + (ex - sx) * t) * (w - cw))
# #     y0 = int((sy + (ey - sy) * t) * (h - ch))

# #     x0 = max(0, min(x0, w - cw))
# #     y0 = max(0, min(y0, h - ch))

# #     return img.crop((x0, y0, x0 + cw, y0 + ch))

# # def main():
# #     paths = list_frames(FRAMES_DIR)
# #     if not paths:
# #         raise RuntimeError(f"No PNG frames found in: {FRAMES_DIR.resolve()}")

# #     n = len(paths)
# #     fps = max(1, int(round(n / DURATION_SEC)))  # choose fps so total ~30 sec

# #     # To guarantee ~30 sec exactly, we resample frames to N_out = fps * duration
# #     n_out = int(round(fps * DURATION_SEC))
# #     idx = np.linspace(0, n - 1, n_out).round().astype(int)

# #     print(f"[INFO] Found {n} frames")
# #     print(f"[INFO] Writing {n_out} frames at {fps} fps => ~{DURATION_SEC}s")
# #     print(f"[INFO] Output: {OUT_MP4.resolve()}")

# #     with imageio.get_writer(OUT_MP4, fps=fps, codec="libx264") as writer:
# #         for k, i in enumerate(idx):
# #             p = paths[i]
# #             img = Image.open(p).convert("RGB")

# #             if DO_PAN:
# #                 t = 0.0 if n_out <= 1 else (k / (n_out - 1))
# #                 img = crop_pan(img, t)

# #             writer.append_data(np.array(img))

# #     print("[OK] Done.")

# # if __name__ == "__main__":
# #     main()
# import os, re
# from pathlib import Path

# import numpy as np
# import imageio
# from PIL import Image, ImageOps, ImageDraw, ImageFont

# # -----------------------------
# # SETTINGS (edit these)
# # -----------------------------
# FRAMES_DIR   = Path(r"C:\Users\Adrija\Downloads\DFGCN\out_frames_irma")          # folder with your PNGs
# OUT_MP4      = Path(r"storm_noaa_style_30s.mp4")
# DURATION_SEC = 30.0

# # Make it look "NOAA-clean"
# AUTO_CROP_WHITE_MARGINS = True   # removes big white borders
# CROP_EXTRA_PX = 0                # add more cropping if needed (try 8/12)

# # Optional: add timestamp text (from filename or index)
# ADD_TIMESTAMP = True
# TIMESTAMP_MODE = "index"         # "index" or "filename"
# TIMESTAMP_PREFIX = "t = "        # e.g., "Valid: " if you want

# # Optional: smooth pan/zoom (looks like tracking the storm)
# DO_PAN_ZOOM = True
# # Crop window size in pixels (smaller -> more zoom). Set None to disable zoom.
# CROP_W, CROP_H = 960, 540        # for 1280x720 sources this is a nice zoom
# PAN_START = (0.08, 0.52)         # (x_frac, y_frac) top-left start position
# PAN_END   = (0.38, 0.28)         # (x_frac, y_frac) top-left end position

# # Output FPS chosen automatically so total is ~30s
# # -----------------------------


# def natural_sort_key(name: str):
#     nums = re.findall(r"\d+", name)
#     return int(nums[-1]) if nums else 0


# def list_pngs(folder: Path):
#     if not folder.exists():
#         raise FileNotFoundError(
#             f"\n[ERROR] Folder not found:\n  {folder.resolve()}\n\n"
#             f"Fix: Create it OR set FRAMES_DIR to your real frames folder.\n"
#         )
#     files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
#     files.sort(key=natural_sort_key)
#     paths = [folder / f for f in files]
#     if not paths:
#         raise RuntimeError(f"\n[ERROR] No PNG files found in: {folder.resolve()}\n")
#     return paths


# def autocrop_white(img: Image.Image, extra_px: int = 0):
#     """
#     Auto-crop near-white borders (typical 'plot margin' area).
#     Works best when margins are white/light gray.
#     """
#     # Convert to grayscale and invert so content becomes bright
#     gray = img.convert("L")
#     # Treat almost-white as background
#     # Lower threshold if your background isn't pure white
#     bw = gray.point(lambda p: 0 if p > 245 else 255, mode="1")
#     bbox = bw.getbbox()
#     if bbox is None:
#         return img  # nothing to crop

#     left, upper, right, lower = bbox
#     left  = max(0, left - extra_px)
#     upper = max(0, upper - extra_px)
#     right = min(img.width,  right + extra_px)
#     lower = min(img.height, lower + extra_px)

#     return img.crop((left, upper, right, lower))


# def apply_pan_zoom(img: Image.Image, t: float):
#     """
#     t in [0,1]. Crops a moving window for a smooth pan/zoom effect.
#     """
#     if not DO_PAN_ZOOM or CROP_W is None or CROP_H is None:
#         return img

#     w, h = img.size
#     cw, ch = min(CROP_W, w), min(CROP_H, h)

#     sx, sy = PAN_START
#     ex, ey = PAN_END

#     # top-left moves across the image
#     x0 = int((sx + (ex - sx) * t) * (w - cw))
#     y0 = int((sy + (ey - sy) * t) * (h - ch))

#     x0 = max(0, min(x0, w - cw))
#     y0 = max(0, min(y0, h - ch))

#     return img.crop((x0, y0, x0 + cw, y0 + ch))


# def add_timestamp(img: Image.Image, text: str):
#     if not ADD_TIMESTAMP:
#         return img

#     img = img.copy()
#     draw = ImageDraw.Draw(img)

#     # Try a decent default font (falls back safely)
#     try:
#         font = ImageFont.truetype("arial.ttf", 24)
#     except:
#         font = ImageFont.load_default()

#     pad = 10
#     # Measure text box
#     bbox = draw.textbbox((0, 0), text, font=font)
#     tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

#     # Draw semi-opaque rectangle (simulate with solid dark box)
#     x, y = pad, pad
#     rect = (x - 6, y - 6, x + tw + 10, y + th + 10)
#     draw.rectangle(rect, fill=(0, 0, 0))

#     # Draw text
#     draw.text((x, y), text, fill=(255, 255, 255), font=font)
#     return img


# def main():
#     paths = list_pngs(FRAMES_DIR)

#     n_in = len(paths)
#     fps = max(1, int(round(n_in / DURATION_SEC)))
#     n_out = int(round(fps * DURATION_SEC))

#     # resample indices so output duration is ~30s
#     idx = np.linspace(0, n_in - 1, n_out).round().astype(int)

#     print(f"[INFO] Frames found: {n_in}")
#     print(f"[INFO] Output frames: {n_out} @ {fps} fps (~{DURATION_SEC}s)")
#     print(f"[INFO] Writing: {OUT_MP4.resolve()}")

#     with imageio.get_writer(OUT_MP4, fps=fps, codec="libx264") as writer:
#         for k, i in enumerate(idx):
#             p = paths[i]
#             img = Image.open(p).convert("RGB")

#             # NOAA-clean: remove plot margins
#             if AUTO_CROP_WHITE_MARGINS:
#                 img = autocrop_white(img, extra_px=CROP_EXTRA_PX)

#             # Smooth pan/zoom across the map
#             t = 0.0 if n_out <= 1 else (k / (n_out - 1))
#             img = apply_pan_zoom(img, t)

#             # Timestamp label
#             if ADD_TIMESTAMP:
#                 if TIMESTAMP_MODE == "filename":
#                     stamp = p.stem
#                 else:
#                     stamp = f"{k+1}/{n_out}"
#                 img = add_timestamp(img, f"{TIMESTAMP_PREFIX}{stamp}")

#             writer.append_data(np.array(img))

#             if (k + 1) % 60 == 0:
#                 print(f"  wrote {k+1}/{n_out}")

#     print("[OK] Done.")


# if __name__ == "__main__":
#     main()
import os
import re
import tarfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import imageio
from PIL import Image
from ftplib import FTP


# =========================================================
# SETTINGS
# =========================================================
RADAR_PAGE_URL = "https://www.aoml.noaa.gov/hrd/Storm_pages/irma2017/radar.html"
OUT_DIR = Path("aoml_download")
OUT_MP4 = Path("irma_aoml_noaa_style_30s.mp4")

DURATION_SEC = 30
FPS = 24

# Make the frames "NOAA clean"
AUTO_CROP_WHITE_MARGINS = True
CROP_EXTRA_PX = 6

# Pick which FTP "single frame GIFs" folder to use (0 = first one on page)
FOLDER_INDEX = 0

# If a folder contains multiple .tar.gz/.tgz bundles, choose which
TARBALL_INDEX = 0
# =========================================================


def autocrop_white(img: Image.Image, extra_px: int = 0) -> Image.Image:
    gray = img.convert("L")
    bw = gray.point(lambda p: 0 if p > 245 else 255, mode="1")
    bbox = bw.getbbox()
    if bbox is None:
        return img
    l, u, r, d = bbox
    l = max(0, l - extra_px)
    u = max(0, u - extra_px)
    r = min(img.width, r + extra_px)
    d = min(img.height, d + extra_px)
    return img.crop((l, u, r, d))


def find_ftp_singleframe_folders(page_url: str):
    """
    The AOML radar page includes multiple FTP links for:
    'tar'd and gzipped single frame GIFs' (folders). :contentReference[oaicite:1]{index=1}
    We extract those ftp:// links.
    """
    html = requests.get(page_url, timeout=60).text

    # grab all ftp links
    ftp_links = re.findall(r'href="(ftp://[^"]+)"', html, flags=re.IGNORECASE)

    # keep only the ones whose anchor text mentions "single frame GIFs"
    # since we don't have parsed DOM, we approximate by scanning nearby text
    hits = []
    for m in re.finditer(r'href="(ftp://[^"]+)"', html, flags=re.IGNORECASE):
        url = m.group(1)
        window = html[max(0, m.start()-120): m.end()+120].lower()
        if "single frame" in window and "gif" in window:
            hits.append(url)

    # dedupe preserve order
    def dedupe(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    hits = dedupe(hits)
    if not hits:
        # fallback: return all ftp links if our text-window filter misses
        hits = dedupe(ftp_links)

    return hits


def ftp_list_tarballs(ftp_url: str):
    """
    Given an ftp://host/path/ URL, list .tar.gz / .tgz files inside.
    """
    u = urlparse(ftp_url)
    host = u.hostname
    path = u.path

    ftp = FTP(host, timeout=60)
    ftp.login()         # anonymous
    ftp.set_pasv(True)  # usually required behind NAT

    ftp.cwd(path)

    files = ftp.nlst()
    tarballs = [f for f in files if f.lower().endswith((".tar.gz", ".tgz"))]

    ftp.quit()
    tarballs.sort()
    return host, path, tarballs


def ftp_download(host: str, folder_path: str, filename: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ftp = FTP(host, timeout=120)
    ftp.login()
    ftp.set_pasv(True)
    ftp.cwd(folder_path)

    print(f"[INFO] FTP download: ftp://{host}{folder_path}/{filename}")
    with open(out_path, "wb") as f:
        ftp.retrbinary(f"RETR {filename}", f.write, blocksize=1024 * 64)

    ftp.quit()
    return out_path


def extract_tar_gz(tar_path: Path, out_folder: Path):
    out_folder.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Extracting: {tar_path.name} -> {out_folder.resolve()}")
    mode = "r:gz" if tar_path.name.lower().endswith(".tar.gz") else "r:*"
    with tarfile.open(tar_path, mode) as tf:
        tf.extractall(out_folder)
    return out_folder


def gather_frames(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".gif"}
    paths = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    if not paths:
        raise RuntimeError(f"[ERROR] No image frames found in {folder.resolve()}")

    # sort by numbers in filename, else by name
    def key(p: Path):
        nums = re.findall(r"\d+", p.stem)
        return (int(nums[-1]) if nums else 10**18, p.name.lower())

    paths.sort(key=key)
    return paths


def write_30s_mp4(frame_paths, out_mp4: Path, fps: int, duration_sec: int):
    n_out = fps * duration_sec
    idx = np.linspace(0, len(frame_paths) - 1, n_out).round().astype(int)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing MP4: {out_mp4.resolve()}")
    print(f"[INFO] Using {len(frame_paths)} source frames -> {n_out} frames @ {fps} fps (~{duration_sec}s)")

    with imageio.get_writer(out_mp4, fps=fps, codec="libx264") as writer:
        for k, i in enumerate(idx):
            img = Image.open(frame_paths[i]).convert("RGB")
            if AUTO_CROP_WHITE_MARGINS:
                img = autocrop_white(img, extra_px=CROP_EXTRA_PX)
            writer.append_data(np.array(img))

            if (k + 1) % 120 == 0:
                print(f"  wrote {k+1}/{n_out}")

    print("[OK] Done.")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    folders = find_ftp_singleframe_folders(RADAR_PAGE_URL)
    if not folders:
        raise RuntimeError("[ERROR] Could not find any FTP folders on the AOML radar page.")

    pick_folder = min(FOLDER_INDEX, len(folders) - 1)
    ftp_folder_url = folders[pick_folder]

    host, path, tarballs = ftp_list_tarballs(ftp_folder_url)
    if not tarballs:
        raise RuntimeError(
            f"[ERROR] No .tar.gz/.tgz found in:\n  {ftp_folder_url}\n"
            "This can happen if the FTP folder contains raw GIFs instead of tarballs."
        )

    pick_tar = min(TARBALL_INDEX, len(tarballs) - 1)
    tar_name = tarballs[pick_tar]

    tar_path = OUT_DIR / tar_name
    ftp_download(host, path, tar_name, tar_path)

    extracted = OUT_DIR / "frames_extracted"
    extract_tar_gz(tar_path, extracted)

    frames = gather_frames(extracted)
    write_30s_mp4(frames, OUT_MP4, FPS, DURATION_SEC)


if __name__ == "__main__":
    main()


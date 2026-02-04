import re
import tarfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import imageio
from PIL import Image
from bs4 import BeautifulSoup
from ftplib import FTP


# =========================
# SETTINGS
# =========================
RADAR_PAGE_URL = "https://www.aoml.noaa.gov/hrd/Storm_pages/irma2017/radar.html"

OUT_DIR = Path("aoml_download")
OUT_MP4 = Path("irma_aoml_30s.mp4")

DURATION_SEC = 30
FPS = 24

AUTO_CROP_WHITE_MARGINS = True
CROP_EXTRA_PX = 6

# Choose which link on the page (0 = first single-frame GIF link)
LINK_INDEX = 0

# If the link is a folder containing many tarballs, pick which one
TARBALL_INDEX = 0
# =========================


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


def normalize_to_ftp(href: str) -> str:
    href = href.strip()
    if href.startswith("//"):
        href = "https:" + href
    if not re.match(r"^[a-zA-Z]+://", href):
        href = "https://" + href
    u = urlparse(href)
    return f"ftp://{u.netloc}{u.path}"


def find_singleframe_links(page_url: str):
    """
    Find links on the AOML page whose anchor text mentions single-frame GIFs.
    These may point directly to a .tar.gz file or a folder. ([aoml.noaa.gov](https://www.aoml.noaa.gov/hrd/Storm_pages/irma2017/radar.html))
    """
    html = requests.get(page_url, timeout=60).text
    soup = BeautifulSoup(html, "lxml")

    hits = []
    for a in soup.find_all("a"):
        text = (a.get_text() or "").strip().lower()
        href = (a.get("href") or "").strip()
        if not href:
            continue

        if ("single frame" in text) and ("gif" in text) and ("ftp.aoml.noaa.gov" in href.lower()):
            hits.append(normalize_to_ftp(href))

    # dedupe preserve order
    seen = set()
    out = []
    for x in hits:
        if x not in seen:
            seen.add(x)
            out.append(x)

    return out


def ftp_connect(host: str, timeout=180):
    ftp = FTP(host, timeout=timeout)
    ftp.login()         # anonymous
    ftp.set_pasv(True)
    return ftp


def ftp_download_file(ftp_url: str, out_path: Path):
    """
    Download a single file from an ftp:// URL.
    """
    u = urlparse(ftp_url)
    host = u.hostname
    file_path = u.path
    folder = str(Path(file_path).parent).replace("\\", "/")
    filename = Path(file_path).name

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ftp = ftp_connect(host)
    ftp.cwd(folder)

    print(f"[INFO] Downloading file:\n  {ftp_url}\n  -> {out_path.resolve()}")
    with open(out_path, "wb") as f:
        ftp.retrbinary(f"RETR {filename}", f.write, blocksize=1024 * 64)

    ftp.quit()
    return out_path


def ftp_list_tarballs_in_folder(ftp_folder_url: str):
    """
    List .tar.gz/.tgz files in an FTP folder URL.
    """
    u = urlparse(ftp_folder_url)
    host = u.hostname
    folder_path = u.path

    ftp = ftp_connect(host)
    ftp.cwd(folder_path)

    files = ftp.nlst()
    tarballs = [f for f in files if f.lower().endswith((".tar.gz", ".tgz"))]
    tarballs.sort()

    ftp.quit()
    return host, folder_path, tarballs


def extract_tarball(tar_path: Path, out_folder: Path):
    out_folder.mkdir(parents=True, exist_ok=True)
    mode = "r:gz" if tar_path.name.lower().endswith(".tar.gz") else "r:*"

    print(f"[INFO] Extracting: {tar_path.name} -> {out_folder.resolve()}")
    with tarfile.open(tar_path, mode) as tf:
        tf.extractall(out_folder)

    return out_folder


def gather_frames(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".gif"}
    paths = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    if not paths:
        raise RuntimeError(f"[ERROR] No image frames found in: {folder.resolve()}")

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
    print(f"[INFO] Source frames: {len(frame_paths)} -> Output: {n_out} frames @ {fps} fps (~{duration_sec}s)")

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

    links = find_singleframe_links(RADAR_PAGE_URL)
    if not links:
        raise RuntimeError("[ERROR] Could not find single-frame GIF links on the AOML page.")

    link = links[min(LINK_INDEX, len(links) - 1)]
    print(f"[INFO] Using link: {link}")

    extracted = OUT_DIR / "frames_extracted"

    # Case 1: link is already a tarball file
    if link.lower().endswith((".tar.gz", ".tgz")):
        tar_path = OUT_DIR / Path(urlparse(link).path).name
        ftp_download_file(link, tar_path)
        extract_tarball(tar_path, extracted)

    # Case 2: link is a folder, list tarballs inside
    else:
        host, folder_path, tarballs = ftp_list_tarballs_in_folder(link)
        if not tarballs:
            raise RuntimeError(f"[ERROR] No .tar.gz/.tgz found in FTP folder:\n  {link}")

        tar_name = tarballs[min(TARBALL_INDEX, len(tarballs) - 1)]
        tar_url = f"ftp://{host}{folder_path.rstrip('/')}/{tar_name}"
        tar_path = OUT_DIR / tar_name

        ftp_download_file(tar_url, tar_path)
        extract_tarball(tar_path, extracted)

    frames = gather_frames(extracted)
    write_30s_mp4(frames, OUT_MP4, FPS, DURATION_SEC)


if __name__ == "__main__":
    main()

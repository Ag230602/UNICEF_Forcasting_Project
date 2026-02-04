import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# =====================================================
# CONFIG
# =====================================================
MAP_DIR = r"C:\Users\Adrija\Downloads\DFGCN\outputs"
MAP_FILENAME = None  # put exact filename if you want (e.g. "world_map.png")

OUT_PATH = os.path.join(MAP_DIR, "WORLD_MAP_TRACKS_FINAL.png")

# World map extent (must match the image)
MAP_EXTENT = (-180, 180, -90, 90)

SHOW_ARROWS = True
SHOW_TIME_LABELS = True
TIME_EVERY = 1
DX, DY = 2.5, 1.5


# =====================================================
# GET WORLD MAP IMAGE (ONCE)
# =====================================================
def get_map_image():
    if MAP_FILENAME:
        path = os.path.join(MAP_DIR, MAP_FILENAME)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path

    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(glob.glob(os.path.join(MAP_DIR, ext)))

    if not files:
        raise FileNotFoundError("Put a world map image in outputs folder")

    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


# =====================================================
# DRAW TRACK (ONCE)
# =====================================================
def draw_track(ax, latlon, color, label, lead_hours):
    lat = latlon[:, 0]
    lon = latlon[:, 1]

def draw_track(ax, latlon, color, label, lead_hours):
    lat = latlon[:, 0]
    lon = latlon[:, 1]

    # legend handle (dummy, no line drawn)
    ax.plot([], [], color=color, lw=3, marker="o", label=label)

    # arrows only (actual path)
    for i in range(len(lon) - 1):
        ax.annotate(
            "",
            xy=(lon[i + 1], lat[i + 1]),
            xytext=(lon[i], lat[i]),
            arrowprops=dict(arrowstyle="->", lw=2, color=color),
            zorder=6,
        )

    # time labels
    for i in range(len(lead_hours)):
        ax.text(
            lon[i] + DX,
            lat[i] + DY,
            f"{int(lead_hours[i])}h",
            fontsize=9,
            weight="bold",
            color=color,
            zorder=7,
        )
    

    if SHOW_ARROWS:
        for i in range(len(lon) - 1):
            ax.annotate(
                "",
                xy=(lon[i + 1], lat[i + 1]),
                xytext=(lon[i], lat[i]),
                arrowprops=dict(arrowstyle="->", lw=2, color=color),
                zorder=6,
            )

    if SHOW_TIME_LABELS:
        for i in range(0, len(lead_hours), TIME_EVERY):
            ax.text(
                lon[i] + DX,
                lat[i] + DY,
                f"{int(lead_hours[i])}h",
                fontsize=9,
                weight="bold",
                color=color,
                zorder=7,
            )


# =====================================================
# MAIN (RUNS ONCE)
# =====================================================
def main():
    bg_path = get_map_image()
    img = mpimg.imread(bg_path)

    fig, ax = plt.subplots(figsize=(15, 8))

    # ---- WORLD MAP BACKGROUND (ONCE ONLY) ----
    ax.imshow(
        img,
        extent=MAP_EXTENT,
        origin="upper",
        aspect="auto",
        zorder=0,
    )

    ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])

    # ---- SAMPLE TRACK DATA (REPLACE WITH YOUR REAL OUTPUTS) ----
    lead_hours = np.array([0, 6, 12, 24, 48])

    actual = np.array([
        [15, -30],
        [17, -40],
        [19, -50],
        [22, -60],
        [26, -70],
    ])

    lstm = actual + np.array([[0,0],[0.3,1],[0.4,2],[0.5,3],[0.6,4]])
    transformer = actual + np.array([[0,0],[1.2,3],[2.0,5],[2.8,7],[3.5,9]])
    gno = actual + np.array([[0,0],[0.2,0.5],[0.3,1],[0.3,1.4],[0.4,1.8]])

    # ---- DRAW TRACKS (ONCE EACH) ----
    draw_track(ax, actual, "cyan", "Actual", lead_hours)
    draw_track(ax, lstm, "orange", "LSTM", lead_hours)
    draw_track(ax, transformer, "lime", "Transformer", lead_hours)
    draw_track(ax, gno, "red", "GNO+DynGNN", lead_hours)

    # ---- STYLE ----
    ax.grid(True, linestyle=":", color="deepskyblue", alpha=0.7)
    ax.set_title("World Map Background + Actual vs Predicted Tracks")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    plt.close()

    print("✔ Background:", bg_path)
    print("✔ Saved:", OUT_PATH)


# =====================================================
# ENTRY POINT (ONLY ONCE)
# =====================================================
if __name__ == "__main__":
    main()

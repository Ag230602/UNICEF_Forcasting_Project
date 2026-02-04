from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

def make_demo_nc(out_nc: Path, start="2017-09-07", steps=121, freq="30min"):
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    lats = np.linspace(18, 32, 120)
    lons = np.linspace(-88, -70, 160)
    times = pd.date_range(start, periods=steps, freq=freq)  # 121 steps @ 30min ~ 60 minutes? actually 60h

    Lon, Lat = np.meshgrid(lons, lats)

    u = np.zeros((len(times), len(lats), len(lons)), dtype=np.float32)
    v = np.zeros_like(u)

    for ti in range(len(times)):
        eye_lon = -82 + (ti / len(times)) * 10.0   # move east
        eye_lat = 24 + (ti / len(times)) * 6.0    # move north

        dx = (Lon - eye_lon) * np.cos(np.deg2rad(eye_lat))
        dy = (Lat - eye_lat)
        r2 = dx**2 + dy**2 + 1e-3

        strength = 45 * np.exp(-r2 / 6.0)         # peak winds
        u[ti] = (-dy / np.sqrt(r2)) * strength
        v[ti] = ( dx / np.sqrt(r2)) * strength

    ds = xr.Dataset(
        {"u10": (("time","latitude","longitude"), u),
         "v10": (("time","latitude","longitude"), v)},
        coords={"time": times, "latitude": lats, "longitude": lons}
    )
    ds.to_netcdf(out_nc)
    print("Wrote:", out_nc)

def make_demo_track(out_csv: Path, start="2017-09-07", steps=121, freq="30min"):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    times = pd.date_range(start, periods=steps, freq=freq)
    lons = np.linspace(-82, -72, steps)
    lats = np.linspace(24, 30, steps)
    df = pd.DataFrame({"time": times, "lat": lats, "lon": lons})
    df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

if __name__ == "__main__":
    make_demo_nc(Path("data/irma_era5_u10v10.nc"))
    make_demo_track(Path("data/irma_track.csv"))

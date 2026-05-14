from lib import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import numpy as np
import scienceplots
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing
import contextlib
import os

plt.style.use(['science'])

SNAPBASE = 'snap_'
SNAP_START = 1
SNAP_END = 10

OUTDIR = Path('/home/dc-naza3/rds/rds-dirac-dp317-rvYpA2WHqGs/rion/snap-plotting/tests/ccomp')
FRAMES_DIR = OUTDIR 
MOVIE_MP4 = OUTDIR / f'radial_compare_{SNAP_START:03d}_{SNAP_END:03d}.mp4'
MOVIE_GIF = OUTDIR / f'radial_compare_{SNAP_START:03d}_{SNAP_END:03d}.gif'


def build_movie(frame_paths, fps=3):
    frame_paths = sorted(frame_paths)
    first_img = plt.imread(frame_paths[0])

    fig, ax = plt.subplots(figsize=(first_img.shape[1] / 150, first_img.shape[0] / 150), dpi=150)
    im = ax.imshow(first_img)
    ax.axis('off')
    fig.tight_layout(pad=0)

    def update(i):
        im.set_data(plt.imread(frame_paths[i]))
        return (im,)

    anim = FuncAnimation(fig, update, frames=len(frame_paths), interval=1000 / fps, blit=True)

    if writers.is_available('ffmpeg'):
        anim.save(MOVIE_MP4, writer='ffmpeg', fps=fps)
        print(f"Saved movie: {MOVIE_MP4}")
    else:
        print("MovieWriter ffmpeg unavailable; saving GIF with Pillow.")
        anim.save(MOVIE_GIF, writer='pillow', fps=fps)
        print(f"Saved movie: {MOVIE_GIF}")

    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    snapnums = list(range(SNAP_START, SNAP_END + 1))
    frame_paths_expected = [FRAMES_DIR / f'radial_compare_snap{snapnum:03d}.png' for snapnum in snapnums]

    frame_paths = [str(frame_path) for frame_path in frame_paths_expected if frame_path.exists()]
    if len(frame_paths) != len(snapnums):
        missing = [f'radial_compare_snap{snapnum:03d}.png' for snapnum, frame_path in zip(snapnums, frame_paths_expected) if not frame_path.exists()]
        raise RuntimeError(f"Missing frames after generation: {missing}")

    build_movie(frame_paths, fps=3)


if __name__ == '__main__':
    main()
import os
import argparse
import numpy as np
import mdtraj as md
from afmfold.images import generate_images

def parse_args():
    parser = argparse.ArgumentParser(description="Parse arguments for training AFM image generator.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument('--save-dir', type=str, required=True, help='Directory where output images will be saved.')
    parser.add_argument('--pdb-path', type=str, required=True, help='Path to the reference PDB file.')
    parser.add_argument('--dcd-path', type=str, required=True, help='Path to the DCD trajectory file.')
    parser.add_argument('--distance-path', type=str, required=True, help='Path to the precomputed distance numpy file.')
    parser.add_argument('--reference-image-path', type=str, default="", help='Path for reference images.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs.')
    parser.add_argument('--dataset-size', type=int, default=10000, help='Number of samples in the dataset.')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--width', type=int, default=35, help='Width of generated images.')
    parser.add_argument('--height', type=int, default=35, help='Height of generated images.')
    parser.add_argument('--resolution-nm', type=float, default=0.98, help='Spatial resolution in nanometers.')
    parser.add_argument('--noise-nm', type=float, default=0.1, help='Noise in nanometers.')
    parser.add_argument('--min-tip-radius', type=float, default=6.0, help='Minimum tip radius.')
    parser.add_argument('--max-tip-radius', type=float, default=12.0, help='Maximum tip radius.')
    parser.add_argument('--min-tip-angle', type=float, default=10.0, help='Minimum tip angle.')
    parser.add_argument('--max-tip-angle', type=float, default=30.0, help='Minimum tip angle.')
    parser.add_argument('--min-z', type=float, default=0.0, help='Minimum z-coordinate of conformations')
    parser.add_argument('--is-tqdm', action='store_true', help='Whether to show progress bar using tqdm.')
    parser.add_argument('--match-histgram', action='store_true', help='Whether to match histogram of images.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    traj = md.load(args.dcd_path, top=args.pdb_path)
    distance = np.load(args.distance_path)
    
    if os.path.exists(args.reference_image_path):
        ref_images = np.load(args.reference_image_path)
    else:
        ref_images = None
        
    generate_images(
        traj, args.resolution_nm, args.width, args.height, args.epochs, args.dataset_size, 
        distance=distance, batch_size=args.batch_size, min_z=args.min_z, noise_nm=args.noise_nm,
        max_tip_radius=args.max_tip_radius, min_tip_radius=args.min_tip_radius, 
        max_tip_angle=args.max_tip_angle, min_tip_angle=args.min_tip_angle,
        ref_images=ref_images, is_tqdm=args.is_tqdm, match_histgram=args.match_histgram, 
        save_dir=args.save_dir, device=args.device,
        )
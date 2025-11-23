import os
import sys
#sys.modules["deepspeed"] = None
import argparse
import torch
import math
import numpy as np
import mdtraj as md
import glob
from pprint import pprint

from afmfold.cnn import CNSteerableCNN
from afmfold.domain import compute_domain_distance, get_domain_pairs
from afmfold.utils import load_json
from afmfold.inference import inference

def parse_args():
    parser = argparse.ArgumentParser(description="AFM image → CV space → AFM-Fold inference")

    # Image generation
    parser.add_argument("--image-path", type=str, required=True, help="Path of the AFM images.")
    parser.add_argument("--label-path", type=str, default="", help="Path of the CV coordinates corresponding to the AFM images.")
    parser.add_argument("--dcd-path", type=str, default="", help="Path of the trajectory from which the AFM images were generated.")
    parser.add_argument("--pdb-path", type=str, default="", help="Path of the trajectory from which the AFM images were generated.")
    parser.add_argument("--name", type=str, default="", help="Name of the protein")

    # Model / inference
    parser.add_argument("--ckpt", type=str, required=True, help="CNSteerableCNN checkpoint")
    parser.add_argument("--json-path", type=str, required=True, help="Json file with the MSA path")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to output")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--inference-batchsize", type=int, default=32)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--max-trial", type=int, default=5)
    parser.add_argument("--mse-threshold", type=float, default=10.0, help="in Å^2")
    parser.add_argument("--prev-inference-dir", type=str, default="")
    parser.add_argument("--in-nm", action="store_true")

    args = parser.parse_args()
    return args

def gather_restraint_presictions(prev_inference_dir):
    if not os.path.exists(prev_inference_dir):
        return None, None
    
    jsons = glob.glob(os.path.join(prev_inference_dir, "*", "predictions", "*.json"))
    restraint_list = []
    prediction_list = []
    for json in jsons:
        inference_dict = load_json(json)
        if "restraint" in inference_dict and "prediction" in inference_dict:
            restraint_arr = np.array(inference_dict["restraint"])
            prediction_arr = np.array(inference_dict["prediction"])
            restraint_list.append(restraint_arr[None, :])
            prediction_list.append(prediction_arr[None, :])
    prev_restraints = np.concatenate(restraint_list, axis=0)
    prev_predictions = np.concatenate(prediction_list, axis=0)
    return prev_restraints, prev_predictions


def main(args):
    if os.path.isdir(args.image_path):
        image_paths = glob.glob(os.path.join(args.image_path, "image*.npy"))
        image_list = []
        for image_path in image_paths:
            _image = np.load(image_path)
            H, W = _image.shape[-2:]
            image_list.append(_image.reshape((-1, H, W)))
        images = np.concatenate(image_list, axis=0)
    elif args.image_path.endswith(".npy"):
        images = np.load(args.image_path)
    else:
        raise NotImplementedError(args.image_path)
    
    if os.path.exists(args.label_path) and os.path.isdir(args.label_path):
        label_paths = glob.glob(os.path.join(args.label_path, "image*.npy"))
        label_list = []
        for label_path in label_paths:
            _label = np.load(image_path)
            _label_dim = _label.shape[-1]
            label_list.append(_label.reshape((-1, _label_dim)))
        labels = np.concatenate(label_list, axis=0)
    elif os.path.exists(args.label_path) and args.label_path.endswith(".npy"):
        labels = np.load(args.label_path)
    else:
        labels = None
    
    # Extract the corresponding trajectory
    if os.path.exists(args.dcd_path) and os.path.exists(args.pdb_path):
        traj = md.load(args.dcd_path, top=args.pdb_path)
        domain_pairs = get_domain_pairs(args.name)
        ref_distance = np.zeros((len(traj), len(domain_pairs)))
        for i, (d1, d2) in enumerate(domain_pairs):
            ref_distance[:, i] = compute_domain_distance(traj, d1, d2).ravel()
        indices = np.array([np.argmax(np.all(ref_distance == labels[i][None, :], axis=1)) for i in range(len(labels))])
        ref_traj = traj.atom_slice(traj.topology.select("element != H"))[indices]
    elif os.path.exists(args.pdb_path):
        traj = md.load(args.pdb_path)
        domain_pairs = get_domain_pairs(args.name)
        ref_distance = np.zeros((len(traj), len(domain_pairs)))
        for i, (d1, d2) in enumerate(domain_pairs):
            ref_distance[:, i] = compute_domain_distance(traj, d1, d2).ravel()
        indices = np.array([np.argmax(np.all(ref_distance == labels[i][None, :], axis=1)) for i in range(len(labels))])
        ref_traj = traj.atom_slice(traj.topology.select("element != H"))[indices]
    else:
        ref_traj = None
    
    # Load the model
    model, optim, ckpt = CNSteerableCNN.load_from_checkpoint(args.ckpt)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Estimate inter-domain distances from AFM images
    prediction_list = []
    for i in range(math.ceil(len(images) / args.inference_batchsize)):
        images_pt = torch.from_numpy(images[i * args.inference_batchsize : (i + 1) * args.inference_batchsize]).unsqueeze(1).to(device).to(torch.float32)
        out = model(images_pt)
        prediction_list.append(out.detach().cpu().numpy())
    predictions = np.concatenate(prediction_list, axis=0)  # noqa: F841 (used later if needed)

    os.makedirs(args.out_dir, exist_ok=True)

    # Collect previous inference results
    prev_restraints, prev_predictions = gather_restraint_presictions(args.prev_inference_dir)
    
    # Run inference
    cif_list = inference(
        images,
        model,
        args.json_path,
        args.out_dir,
        base_seed=args.base_seed,
        device=device,
        max_trial=args.max_trial,
        mse_threshold=args.mse_threshold,
        labels=labels,
        traj=ref_traj,
        predictions=predictions,
        prev_predictions=prev_predictions,
        prev_restraints=prev_restraints,
        in_nm=args.in_nm,
    )

    pprint(cif_list)


if __name__ == "__main__":
    args = parse_args()
    main(args)

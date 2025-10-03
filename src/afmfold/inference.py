import os
import torch
import numpy as np
import mdtraj as md
import json
import glob
from tqdm import tqdm

from afmfold.runner.batch_inference import inference_jsons
from afmfold.domain import compute_domain_distance, get_domain_pairs
from afmfold.utils import suppress_output, add_arg_to_json, k_nearest_weighted_average

def search_previous_settings(out_dir, is_tqdm=False):
    json_files = glob.glob(os.path.join(out_dir, "*", "predictions", "*.json"))
    restraint_list = []
    prediction_list = []
    for i in tqdm(range(len(json_files)), disable=not is_tqdm):
        with open(json_files[i], mode="r") as f:
            settings = json.load(f)
        if "restraint" in settings and "prediction" in settings:
            restraint_list.append(np.array(settings["restraint"])[None,:])
            prediction_list.append(np.array(settings["prediction"])[None,:])
    restraints = np.concatenate(restraint_list, axis=0)
    predictions = np.concatenate(prediction_list, axis=0)
    return restraints, predictions

def inference(
    images,
    model,
    json_file,
    out_dir,
    base_seed,
    device,
    t_start=0.54,
    y_max=0.60,
    max_trial=3,
    mse_threshold=10.0,
    suppress=True,
    labels=None,
    traj=None,
    predictions=None,
    prev_restraints=None, 
    prev_predictions=None,
):
    # Get the name
    with open(json_file, mode="r") as f:
        data = json.load(f)
    name = data[0]["name"]

    # Initial settings
    if isinstance(device, str):
        device = torch.device(device)
    model.to(device)
    model = model.eval()
    domain_pairs = get_domain_pairs(name)
    H, W = images.shape[-2:]
    images = images.reshape((-1, H, W))
    
    # Run inference
    cif_list = []
    seed = base_seed
    for i in tqdm(range(len(images))):
        # Determine restraints
        if predictions is None:
            images_pt = torch.from_numpy(images[i]).unsqueeze(0).unsqueeze(0).to(torch.float).to(device)
            out = model(images_pt).detach().cpu().numpy()[0]
        else:
            assert len(predictions) == len(images)
            out = predictions[i]
        
        if prev_restraints is not None and prev_predictions is not None:
            target_domain_distance = k_nearest_weighted_average(prev_restraints, prev_predictions, np.copy(out[None,:]), k=6)[0]
        else:
            target_domain_distance = np.copy(out)

        for j in range(max_trial):
            kwargs = {
                "sample_diffusion.N_sample": 1,
                "guidance_kwargs.t_start": t_start,
                "guidance_kwargs.manual": target_domain_distance,
                "guidance_kwargs.scaling_kwargs.func_type": "sigmoid",
                "guidance_kwargs.scaling_kwargs.y_max": y_max,
                "guidance_kwargs.domain_pairs": domain_pairs,
            }

            while len(glob.glob(os.path.join(out_dir, name, f"seed_{seed}"))) > 0:
                seed += 1
            
            seeds = (seed,)
            with suppress_output(suppress=suppress):
                inference_jsons(json_file, out_dir, seeds=seeds, **kwargs)
                
            # Check output
            output_cifs = glob.glob(os.path.join(out_dir, name, f"seed_{seeds[0]}", "predictions", "*.cif"))
            assert len(output_cifs) == 1, output_cifs
            output_cif = output_cifs[0]

            output_jsons = glob.glob(os.path.join(out_dir, name, f"seed_{seeds[0]}", "predictions", "*.json"))
            assert len(output_jsons) == 1, output_jsons
            output_json = output_jsons[0]

            pred_traj = md.load(output_cif)
            pred_domain_distance = np.zeros((len(domain_pairs),))
            for k, (d1, d2) in enumerate(domain_pairs):
                d = compute_domain_distance(pred_traj, d1, d2)
                pred_domain_distance[k] = d.item()
            pred_domain_distance *= 10.0

            # Write restraints into prev_restraints and prev_predictions
            if prev_restraints is not None and prev_predictions is not None:
                prev_restraints = np.concatenate([prev_restraints, target_domain_distance[None,:]], axis=0)
                prev_predictions = np.concatenate([prev_predictions, pred_domain_distance[None,:]], axis=0)
            
            # Append restraints to output
            mse = np.sum((out - pred_domain_distance) ** 2)
            restraint_info = {
                "target": out.tolist(),
                "restraint": target_domain_distance.tolist(),
                "prediction": pred_domain_distance.tolist(),
                "mse": mse.item(),
            }
            
            if labels is not None:
                assert len(labels) == len(images)
                restraint_info["truth"] = (10.0 * labels[i]).tolist()
                
            add_arg_to_json(output_json, restraint_info)

            # Compute differences
            delta = out - pred_domain_distance
            adelta = np.abs(delta)
            mask = adelta > 1.0
            target_domain_distance[mask] = target_domain_distance[mask] + delta[mask]
            
            if mse < mse_threshold:
                # Save other elements
                input_dict = {
                    "image": images[i],
                }
                if labels is not None:
                    assert len(labels) == len(images)
                    input_dict["label"] = labels[i]
                if traj is not None:
                    assert len(traj) == len(images)
                    input_dict["coords"] = traj.xyz[i]
                np.savez_compressed(
                    os.path.join(out_dir, name, f"seed_{seeds[0]}", "predictions", "inputs.npz"), **input_dict
                )

                # Store
                cif_list.append(output_cif)
                break
            else:
                pass

            seed += 1

    return cif_list


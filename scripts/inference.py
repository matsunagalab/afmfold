import os
import sys
sys.modules["deepspeed"] = None
import argparse
import torch
import math
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt  # noqa: F401  (残しておきます)
import json
import glob
from tqdm import tqdm
from pprint import pprint

from afmfold.runner.batch_inference import inference_jsons
from afmfold.cnn import CNSteerableCNN
from afmfold.domain import compute_domain_distance, get_domain_pairs
from afmfold.utils import suppress_output, compute_rmsd_single_frame, k_nearest_weighted_average, add_arg_to_json  # noqa: F401
from afmfold.images import generate_images
from afmfold.visualization import plot_afm  # noqa: F401
from afmfold.rigid_body_fitting import RigidBodyFitting  # noqa: F401

def parse_args():
    parser = argparse.ArgumentParser(description="AFM image → domain distance → AFM-Fold inference")

    # 画像生成まわり
    parser.add_argument("--image-path", type=str, required=True, help="Path of the AFM images.")
    parser.add_argument("--label-path", type=str, default="", help="Path of the CV coodinates corresponding to the AFM images.")
    parser.add_argument("--traj-path", type=str, default="", help="Path of the trajectory from which the AFM images were generated.")
    parser.add_argument("--name", type=str, default="", help="Name of the protein")

    # モデル/推論
    parser.add_argument("--ckpt", type=str, required=True, help="CNSteerableCNN checkpoint")
    parser.add_argument("--json-path", type=str, required=True, help="Json file with the MSA path")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to output")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--inference-batchsize", type=int, default=32)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--max-trial", type=int, default=5)
    parser.add_argument("--mse-threshold", type=float, default=10.0, help="in Å^2")

    args = parser.parse_args()
    return args

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
    max_trial=3,
    mse_threshold=10.0,
    labels=None,
    traj=None,
    predictions=None,
):
    # 名前の取得
    with open(json_file, mode="r") as f:
        data = json.load(f)
    name = data[0]["name"]

    prev_restraints, prev_predictions = search_previous_settings(os.path.join(out_dir, name, "prev"))
    model = model.eval()
    domain_pairs = get_domain_pairs(name)

    cif_list = []
    seed = base_seed
    for i in tqdm(range(len(images))):
        # 拘束を決定
        if predictions is None:
            images_pt = torch.from_numpy(images[i]).unsqueeze(0).unsqueeze(0).to(device)
            out = model(images_pt).detach().cpu().numpy()[0]
        else:
            assert len(predictions) == len(images)
            out = predictions[i]
        
        #target_domain_distance = np.copy(out)
        target_domain_distance = k_nearest_weighted_average(prev_restraints, prev_predictions, np.copy(out[None,:]), k=6)[0]

        for j in range(max_trial):
            kwargs = {
                "sample_diffusion.N_sample": 1,
                "guidance_kwargs.t_start": 0.54,
                "guidance_kwargs.manual": target_domain_distance,
                "guidance_kwargs.scaling_kwargs.func_type": "sigmoid",
                "guidance_kwargs.scaling_kwargs.y_max": 0.60,
                "guidance_kwargs.domain_pairs": domain_pairs,
            }

            while len(glob.glob(os.path.join(out_dir, name, f"seed_{seed}"))) > 0:
                seed += 1

            with suppress_output(True):
                seeds = (seed,)
                inference_jsons(json_file, out_dir, seeds=seeds, **kwargs)

            # 出力の確認
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

            # 拘束をprev_restraints, prev_predictionsに書き込み
            prev_restraints = np.concatenate([prev_restraints, target_domain_distance[None,:]], axis=0)
            prev_predictions = np.concatenate([prev_predictions, pred_domain_distance[None,:]], axis=0)
            
            # 拘束を出力に追加書き込み
            mse = np.sum((out - pred_domain_distance) ** 2)
            restraint_info = {
                "truth": (10.0 * labels[i]).tolist(),
                "target": out.tolist(),
                "restraint": target_domain_distance.tolist(),
                "prediction": pred_domain_distance.tolist(),
                "mse": mse.item(),
            }
            add_arg_to_json(output_json, restraint_info)

            # 差分を計算
            delta = out - pred_domain_distance
            adelta = np.abs(delta)
            mask = adelta > 1.0
            target_domain_distance[mask] = target_domain_distance[mask] + delta[mask]
            
            if mse < mse_threshold:
                # 他の要素を保存
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

                # 格納
                cif_list.append(output_cif)
                break
            else:
                pass

            seed += 1

    return cif_list

def main(args):
    images = np.load(args.image_path)
    
    if os.path.exists(args.label_path):
        labels = np.load(args.label_path)
    else:
        labels = None

    # 対応するトラジェクトリを抜き出す
    if os.path.exists(args.traj_path):
        traj = md.load(args.traj_path)
        domain_pairs = get_domain_pairs(args.name)
        ref_distance = np.zeros((len(traj), len(domain_pairs)))
        for i, (d1, d2) in domain_pairs:
            ref_distance[:,i] = compute_domain_distance(traj, d1, d2)
        indices = np.array([np.argmax(np.all(ref_distance == labels[i][None, :], axis=1)) for i in range(len(labels))])
        ref_traj = traj.atom_slice(traj.topology.select("element != H"))[indices]
    else:
        ref_traj = None
    
    # モデルの読み込み
    model, optim, ckpt = CNSteerableCNN.load_from_checkpoint(args.ckpt)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 画像からドメイン間距離を推定
    prediction_list = []
    for i in range(math.ceil(len(images) / args.inference_batchsize)):
        images_pt = torch.from_numpy(images[i * args.inference_batchsize : (i + 1) * args.inference_batchsize]).unsqueeze(1).to(device)
        out = model(images_pt)
        prediction_list.append(out.detach().cpu().numpy())
    predictions = np.concatenate(prediction_list, axis=0)  # noqa: F841  （必要なら後段に利用）

    os.makedirs(args.out_dir, exist_ok=True)

    # 推論
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
    )

    pprint(cif_list)


if __name__ == "__main__":
    args = parse_args()
    main(args)

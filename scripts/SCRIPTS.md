## Training
In AFM-Fold, a CNN is trained to learn the correspondence between AFM images and the collective variables (CVs) of the underlying structures.
The following code examples illustrate how to generate noiseless training data from MD conformations of Adenylate Kinase (with labels of inter-domain distances), and then train the CNN.

1. **Prepare a candidate conformation set** using [generate_candidates.py](generate_candidates.py). 
    
    ```bash
    # Prepare output directories
    mkdir -p data/candidates

    # Generate candidates.
    nohup python scripts/generate_candidates.py \
        --native-pdb storage/ak.pdb \
        --name 4ake \
        --json-path ./storage/4ake-with-msa.json \
        --out-dir data/candidates/ \
        > candidates.log 2>&1 &

    # Check results.
    ls data/candidates/4ake/
    ```

   The exact candidate conformation set used in the paper is distributed as `../results/ak_train.dcd/pdb` (for FlhA_C, `../results/flhac_train.dcd/pdb`).
    Also, if you have a long MD simulation, candidate conformation set can also be extracted from it.

3. **Compute CVs corresponding to the trajectory** 
    To compute inter-domain distances from the MD trajectory, run the following python code.
    
    ```python
    import mdtraj as md
    import numpy as np
    from afmfold.domain import get_domain_pairs, compute_domain_distance

    # Load the candidate conformation set
    traj = md.load("results/ak_train.dcd", top="results/ak_train.pdb")

    # Get domain pairs (list of tuples, each containing two numpy arrays of residue indices)
    domain_pairs = get_domain_pairs("4ake")

    # Compute inter-domain distances
    domain_distance = np.zeros((len(traj), len(domain_pairs)))
    for i, (d1, d2) in enumerate(domain_pairs):
        domain_distance[:, i] = compute_domain_distance(traj, d1, d2).ravel()

    # Convert in Å.
    domain_distance *= 10.0

    # Save the result
    np.save("results/ak_train_distance.npy", domain_distance)
    ```

4. **Generate training data** using [`generate_images.py`](generate_images.py):

    ```bash
    # Prepare output directories
    mkdir -p data/ak

    # Generate AFM images and labels (epochs * dataset_size data points)
    nohup python scripts/generate_images.py \
        --save-dir data/ak/ \
        --pdb-path results/ak_train.pdb \
        --dcd-path results/ak_train.dcd \
        --distance-path results/ak_train_distance.npy \
        --width 35 --height 35 \
        --resolution-nm 0.3 \
        --min-tip-radius 1.0 --max-tip-radius 2.0 \
        --noise-nm 0.0 \
        --device cuda \
        --epochs 500 \
        --dataset-size 10000 \
        > images.log 2>&1 &

    # Check results (image_0-499.npy, label_0-499.npy should be created)
    ls data/ak/
    ```

5. **Train the CNN** using [`train.py`](train.py):

    ```bash
    # Create output directory for trained models
    mkdir -p models/ak

    # Train the CNN
    nohup python scripts/train.py \
        --data-dir data/ak/ \
        --output-root models/ak/ \
        --epochs 500 \
        --device cuda \
        > train.log 2>&1 &
    ```
    After training, the model checkpoints and loss functions will be stored in `models/ak/`.

## Inference
    Inference can be run either in [`../notebooks/example.ipynb`](../notebooks/example.ipynb) or directly via [`inference.py`](inference.py):

    ```bash
    # Generate json file with the MSA path.
    afmfold tojson --input ./storage/4ake.pdb --out_dir ./storage
    afmfold msa --input ./storage/4ake-without-msa.json --out_dir ./storage
    ls ./storage

    # Run inference.
    nohup python scripts/inference.py \
        --name 4ake \
        --image-path path/to/afm_images \
        --ckpt path/to/your_model.pt \
        --json-path ./storage/4ake-with-msa.json \
        --out-dir out/inference/ \
        --device cuda \
        > inference.log 2>&1 &
    ```

## Evaluation
    You can run rigid-body fitting for multiple results.
    For example, if you want to run rigid-body fitting for all inference results `out/inference/4ake/seed_*`, run the following command.

    ```bash
    nohup python scripts/rigid_body_fitting.py \
        --output_dir out/inference/4ake/seed_* \
        --ref-pdb storage/4ake.pdb \
        --resolution-nm 0.3 \
        --name fitting \
        --prove-radius-mean 1.0 \
        --prove-radius-range 1.0 \
        --prove-radius-step 1.0 \
        --steps 50000 \
        --skip-finished \
        --use-ref-structure \  # Remove this when you do not want to run rigid-body fitting with PDB structure. 
        > fitting.log 2>&1 &
    ```

# AFM-Fold

AFM-Fold is an implementation of **protein structure prediction from AFM (Atomic Force Microscopy) images**. 

<!-- ![Example Figure](./figures/ak.png) -->
![Example Figure](./figures/inference.gif)

## Installation

We recommend using Python 3.10–3.12 with CUDA-enabled PyTorch (tested with PyTorch 2.3.1).  
Clone this repository and install dependencies as follows:

```bash
# Clone this repository
git clone https://github.com/matsunagalab/afmfold.git
cd afmfold

# Unzip the released data
wget https://zenodo.org/records/17597490/files/afmfold.zip
unzip results.zip
ls results

# Install dependancies
pip install -e .[e2cnn]
```

## Usage

### Reproducing the Paper Results
The basic usage is demonstrated in [`notebooks/example.ipynb`](notebooks/example.ipynb). This notebook demonstrates the reproduction of the main results from the paper:

- (A) Conditional structure generation (Fig. 1)
- (B) Evaluation of estimation error using MD data (Fig. 2-3)
- (C) Noise robustness (Fig. 3)
- (D) Comparison with rigid-body fitting (Fig. 4) 

### Training & Inference
The procedures for training and inference are described in detail in [`scripts/SCRIPTS.md`](scripts/SCRIPTS.md). 
By following these, you can also reproduce **Fig. 2** of the paper: 
1. Prepare training datasets with different noise levels and train a CNN model on each of them.  
2. Generate pseudo-AFM images from the MD trajectory with diverse noise levels.  
3. Perform inference for all combinations of (1) and (2), and evaluate the results using the evaluation functions explained in [`notebooks/example.ipynb`](notebooks/example.ipynb).

## Citation information
If you use AFM-Fold in your work, please cite as follows:

```bibtex
@article{kawai2025afmfold,
  title   = {Rapid Reconstruction of Atomic 3D Configurations from an AFM image by AlphaFold~3},
  author  = {Tsuyoshi, Kawai and Yasuhiro, Matsunaga},
  journal = {Journal TBD},
  year    = {2025},
  volume  = {xx},
  number  = {yy},
  pages   = {zz-zz},
  doi     = {10.0000/xxxx}
}
```

## Acknowledgements
AFM-Fold relies heavily on the implementation of:  
- [Protenix](https://github.com/bytedance/Protenix/tree/main)  
- [e2cnn](https://github.com/QUVA-Lab/e2cnn)  

## License
AFM-Fold is released under the **MIT License**.  
See the [LICENSE](LICENSE) file for more details.

## Contact Us
If you have any questions, issues, or suggestions, please:  
- Open an issue on GitHub, or  
- Contact us directly via email: [kawai.t.778@ms.saitama-u.ac.jp](mailto:kawai.t.778@ms.saitama-u.ac.jp), [ymatsunaga@mail.saitama-u.ac.jp](mailto:ymatsunaga@mail.saitama-u.ac.jp).

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
wget https://zenodo.org/records/17714204/files/afmfold.zip
unzip afmfold.zip
ls results

# Install dependancies
pip install -e .[e2cnn]
```

## Usage

### Reproducing the Paper Results
The basic usage is demonstrated in [`notebooks/example.ipynb`](notebooks/example.ipynb). This notebook demonstrates the reproduction of the main results from the paper:

- (A) Conditional structure generation
- (B) Evaluation of estimation error using MD data
- (C) Noise robustness
- (D) Comparison with rigid-body fitting
- (E) Guidance scheduling
- (F) Overviewing training data

### Training & Inference
The procedures for training, inference and rigid-body fitting are described in detail in [`scripts/SCRIPTS.md`](scripts/SCRIPTS.md). 

## Citation information
If you use AFM-Fold in your work, please cite as follows:

```bibtex
@article{kawai2025afmfold,
  title   = {AFM-Fold: Rapid Reconstruction of Protein Conformations from AFM Images},
  author  = {Tsuyoshi, Kawai and Yasuhiro, Matsunaga},
  journal = {bioRxiv},
  year    = {2025},
  url     = {https://www.biorxiv.org/content/10.1101/2025.11.17.688836v1},
  doi     = {https://doi.org/10.1101/2025.11.17.688836}
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

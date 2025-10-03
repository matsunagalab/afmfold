# AFM-Fold

AFM-Fold is an implementation of **protein structure prediction from AFM (Atomic Force Microscopy) images**.  
This github page includes:

1. **Pretrained models** used in the paper,
2. **Code to reproduce the results** reported in the paper (excluding real AFM data),
3. **Pseudo-AFM image generation scripts**,
4. **Training scripts**,
5. **Inference scripts**.

<!-- ![Example Figure](./figures/ak.png) -->
![Example Figure](./figures/inference.gif)

## Installation

We recommend using Python 3.10–3.12 with CUDA-enabled PyTorch (tested with PyTorch 2.3.1).  
Clone this repository and install dependencies as follows:

```bash
pip install -e .[e2cnn]
```

## Usage

### Reproducing the Paper Results
The basic usage is demonstrated in `notebooks/example.ipynb`. This notebook illustrates:

- **(A) Fig. 1**: How to generate pseudo-AFM images of the open and closed conformations of Adenylate Kinase (AK), and how to predict 3D structures from each image. 
- **(B) Fig. 2-3**: How to evaluate the statistical accuracy of AFM-Fold’s predictions. 
- **(C) Fig. 3**: How to evaluate the noise robustness of AFM-Fold’s predictions. 
- **(D) Fig. 4**: How to compare the agreement between experimental AFM images of FlhA<sub>C</sub> and pseudo-AFM images, using both AFM-Fold–predicted structures and reference structures obtained from the PDB. 

### Training & Inference
The procedures for training and inference are described in detail in [`scripts/SCRIPTS.md`](scripts/SCRIPTS.md). 
By following these, you can also reproduce **Fig. 2** of the paper: 
1. Prepare training datasets with different noise levels and train a CNN model on each of them.  
2. Generate pseudo-AFM images from the MD trajectory with diverse noise levels.  
3. Perform inference for all combinations of (1) and (2), and evaluate the results using the evaluation functions explained in `notebooks/example.ipynb`.

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
- Contact us directly via email: [kawai.t.778@ms.saitama-u.ac.jp](mailto:kawai.t.778@ms.saitama-u.ac.jp), [ymatsunaga@mail.saitama-u.ac.jp](mailto:ymatsunaga@mail.saitama-u.ac.jp)

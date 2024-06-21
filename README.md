# Cold Diffusion Model for Seismic Signal Denoising

## Some Results
We apply diffusion models for denoising seismograms. Utilizing the STEAD Dataset, we demonstrate the distinct performance of various denoising models:

**Example 1: Qualitative Picker Analysis**
<p align="center">
  <img src="https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake/blob/main/Images/399_direct300.jpg" width="375">
  <img src="https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake/blob/main/Images/399_sampling300.jpg" width="375"> 
</p>
This figure showcases a superior performance of the sampling method in comparison to the direct model. Notably, the direct model exhibits a tendency to retain a certain level of noise prior to the P-wave arrival, which is not observed in the sampling method. This retention of noise in the direct model can lead to premature P-wave picks, as evidenced in the data. Conversely, the sampling method demonstrates a more accurate noise reduction, resulting in a clearer delineation of the P-wave arrival. This comparison underlines the enhanced capability of the sampling method in accurately identifying seismic events, thereby reducing the likelihood of early P-wave detection errors inherent in the direct model approach.

**Example 2: Qualitative Amplitude Analysis**
<p align="center">
  <img src="https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake/blob/main/Images/5737_direct300.jpg" width="375">
  <img src="https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake/blob/main/Images/5737_sampling300.jpg" width="375"> 
</p>
In Example 2, we demonstrate the Cold Diffusion Model's superiority in preserving seismic signal amplitudes through a sampling strategy ('sampling 300'). This method aligns more accurately with the original waveform, maintaining amplitude integrity, as contrasted with the U-Net model ('direct 300') which exhibits amplitude attenuation and increased residuals, particularly in higher amplitude segments.

## Some Possible Applications

**Enhance Automatic Picking Performance**
<p align="center">
  <img src="https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake/blob/main/Images/picker.jpg" width="375">
</p>
In our evaluation, we applied PhaseNet to waveforms for assessing denoiser impacts on P and S wave arrivals. Histograms comparing "direct," "sampling," and "deep denoiser" methods highlighted the "sampling" method's superior accuracy in aligning with manual picks, especially at higher parameter settings. The "direct" method showed improved P-wave accuracy with parameter increases, while the "deep denoiser" displayed moderate recall rates. Overall, S-wave detections were consistently precise across methods, but P-wave picks varied, with the "sampling" method showing the least discrepancy from manual picks. This study underscores the significance of denoising in automated seismic analysis, with the "sampling" approach being notably effective.

## Dataset

**Distribution of STEAD Dataset**
<p align="center">
  <img src="https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake/blob/main/Images/STEAD.png" width="500">
</p>
You can find the dataset used for training, evaluation, and testing on Zenodo: [STEAD subsample 4 CDiffSD](https://zenodo.org/record/10972601)

## How to Start Training

To replicate the results and start training the model, follow these steps:

1. **Setup the Environment**:
   - Ensure you have conda installed. If not, download and install it from [Conda's official website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
   - Download the `environment.yaml` file from this repository.

   ```bash
   conda env create -f environment.yaml
   conda activate cold-diffusion
   ```

2. **Download the Dataset**:
   - Download the STEAD dataset from [Zenodo](https://zenodo.org/record/10972601).

3. **Run the Training Script**:
   - Ensure the dataset is placed in the correct directory as expected by the script.
   - Execute the training script with the following command:

   ```bash
   python train.py --dataset_path path/to/STEAD_dataset
   ```

4. **Monitor Training**:
   - Training logs and checkpoints will be saved in the specified directory. Monitor the training process using these logs.

## Related Paper
* [Seismic Signal Denoising and Decomposition Using Deep Neural Networks](https://arxiv.org/abs/1811.02695)
* [Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise](https://arxiv.org/abs/2208.09392)

## Citation

If you use this code or dataset in your research, please cite our paper:

```bibtex
@article{trappolini2024cold,
  title={Cold diffusion model for seismic denoising},
  author={Trappolini, Daniele and Laurenti, Laura and Poggiali, Giulio and Tinti, Elisa and Galasso, Fabio and Michelini, Alberto and Marone, Chris},
  journal={Journal of Geophysical Research: Machine Learning and Computation},
  volume={1},
  number={2},
  pages={e2024JH000179},
  year={2024},
  publisher={Wiley Online Library}
}
```

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

### Data Loader Usage

To create a DataLoader, use the following parameters:
- `feature_columns`: List of feature columns (e.g., `'Z_channel', 'E_channel', 'N_channel'`).
- `target_columns`: List of target columns (e.g., `'p_arrival_sample', 's_arrival_sample'`).
- `trace_name_column`: Column containing trace names.

The DataLoader has three elements:
- Index for mapping to trace names.
- Normalized or non-normalized channels (`'Z_channel', 'E_channel', 'N_channel'`) based on the normalize flag.
- `'p_arrival_sample'` (P-wave arrival) and `'s_arrival_sample'` (S-wave arrival) for target columns.

This DataLoader setup allows us to train our model using the prepared data.

*Example of how to use* **utils.create_data_loader**:

```python
import pandas as pd
import utils.utils_diff as u

df_path = your_path + 'df_train.csv'
df_path_noise = your_path + 'df_noise_train.csv'

df = pd.read_pickle(df_path)
df_noise = pd.read_pickle(df_path_noise)

feature_columns = ['Z_channel', 'E_channel', 'N_channel']  
target_columns = ['p_arrival_sample', 's_arrival_sample'] 
trace_name_column = 'trace_name' 

train_loader, val_loader, test_loader, index_to_trace_name = u.create_data_loader(df, feature_columns, target_columns, trace_name_column, batch_size= ... , shuffle=True)
train_noise_loader, val_noise_loader, test_noise_loader,index_to_trace_name = u.create_data_loader(df_noise, feature_columns, target_columns, trace_name_column,is_noise = True, batch_size= ..., shuffle=True)
```


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


### Configuration Details

To start training, you need to configure the arguments in the `config_parser.py` file. Here are some important settings:

- Set `training=True` for training mode or `training=False` for testing mode.
- Specify the model path using `path_model` if you want to use a pre-trained model for testing.
- Set the channel type using `channel_type=0` for E, `channel_type=1` for N, and `channel_type=2` for Z.
- During training, you can set `tuning=True` to enable hyperparameter tuning, or `tuning=False` to disable it.
- Adjust the `T` parameter to set the number of timesteps for the model.


## Related Paper
* [Seismic Signal Denoising and Decomposition Using Deep Neural Networks](https://arxiv.org/abs/1811.02695)
* [Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise](https://arxiv.org/abs/2208.09392)
* [Seismic Signal Denoising and Decomposition Using Deep Neural Networks](https://ieeexplore.ieee.org/document/8802278)

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

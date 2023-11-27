# Cold Diffusion Model for seismic signal denoising

The code will be soon released

## Some Results
Apply diffusion models for denoising seismograms. 

We utilize the STEAD Dataset for various denoising models, demonstrating the distinct performance of each model:

#### Example 1: Qualitative Picker Analysis
<p align="center">
  <img src="https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake/blob/main/Images/399_direct300.jpg" width="375">
  <img src="https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake/blob/main/Images/399_sampling300.jpg" width="375"> 
</p>
This figure showcases a superior performance of the sampling method in comparison to the direct model. Notably, the direct model exhibits a tendency to retain a certain level of noise prior to the P-wave arrival, which is not observed in the sampling method. This retention of noise in the direct model can lead to premature P-wave picks, as evidenced in the data. Conversely, the sampling method demonstrates a more accurate noise reduction, resulting in a clearer delineation of the P-wave arrival. This comparison underlines the enhanced capability of the sampling method in accurately identifying seismic events, thereby reducing the likelihood of early P-wave detection errors inherent in the direct model approach.

#### Example 2: Qualitative Amplitude Analysis

<p align="center">
  <img src="https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake/blob/main/Images/5737_direct300.jpg" width="375">
  <img src="https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake/blob/main/Images/5737_sampling300.jpg" width="375"> 
</p>
In Example 2, we demonstrate the Cold Diffusion Model's superiority in preserving seismic signal amplitudes through a sampling strategy ('sampling 300'). This method aligns more accurately with the original waveform, maintaining amplitude integrity, as contrasted with the U-Net model ('direct 300') which exhibits amplitude attenuation and increased residuals, particularly in higher amplitude segments.


**Some Possible Application**
-Enhance automatic picking performance:
<p align="center">
  <img src="https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake/blob/main/Images/picker.jpg" width="375">
</p>
In our evaluation, we applied PhaseNet to waveforms for assessing denoiser impacts on P and S wave arrivals. Histograms comparing "direct," "sampling," and "deep denoiser" methods highlighted the "sampling" method's superior accuracy in aligning with manual picks, especially at higher parameter settings. The "direct" method showed improved P-wave accuracy with parameter increases, while the "deep denoiser" displayed moderate recall rates. Overall, S-wave detections were consistently precise across methods, but P-wave picks varied, with the "sampling" method showing the least discrepancy from manual picks. This study underscores the significance of denoising in automated seismic analysis, with the "sampling" approach being notably effective.
## Dataset

**Distribution of STEAD Dataset**
<p align="center">
  <img src="https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake/blob/main/Images/STEAD.png" width="500">
</p>
You can find the Dataset used for training, evaluation and testing at the following link: https://drive.google.com/drive/folders/1t_jZ0rNiYasJU7_jSrfy-g6aTZ-rkvOu?usp=sharing

## Related Paper
* [Seismic Signal Denoising and Decomposition Using Deep Neural Networks](https://arxiv.org/abs/1811.02695)
* [Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise](https://arxiv.org/abs/2208.09392)

## Other Material (Temporary)
* [Cold Diffusion Material](https://nimble-capri-8e2.notion.site/Cold-Diffusion-b3a6bdce9c2d4c0097aeb814bb86b2ea?pvs=4)

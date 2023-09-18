# Cold Diffusion Model for seismic signal denoising

The code will be soon released

## Some Results
Apply diffusion models for denoising seismograms. 

We apply the STEAD and INSTANCE Dataset for different kind of denoising models, showing the different performance for each model:

**INSTANCE PDF using SI-SDR as metric**



![image](https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake/blob/main/Images/Instance_PDF.jpg)


**STEAD PDF using SI-SDR as metric**





![image](https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake/blob/main/Images/Stead_PDF.jpg)


* One Example of Cold Diffusion Denoising through 500 time step applied on INSTANCE dataset:
![image](https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake/blob/main/Images/Denoised.jpg)

* Examples of Cold Diffusion with small T applied on STEAD dataset:

** T = 5 example 1**


![image](https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake/blob/main/Images/T%3D5.jpg)

** T = 10 example 1**


![image](https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake/blob/main/Images/T%3D10.jpg)

** T = 5 example 2**


![image](https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake/blob/main/Images/T%3D5_2.jpg)

** T = 10 example 2**

![image](https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake/blob/main/Images/T%3D10_2.jpg)

## Dataset

**Distribution of INSTANCE Dataset**
![image](https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake/blob/main/Images/INSTANCE_dataset.png)

**Distribution of STEAD Dataser**
![image](https://github.com/Daniele-Trappolini/Diffusion-Model-for-Earthquake/blob/main/Images/Stead_dataset.png)

You can find the Dataset used for training, evaluation and testing at the following link: https://drive.google.com/drive/folders/1t_jZ0rNiYasJU7_jSrfy-g6aTZ-rkvOu?usp=sharing

## Related Paper
* [Seismic Signal Denoising and Decomposition Using Deep Neural Networks](https://arxiv.org/abs/1811.02695)
* [Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise](https://arxiv.org/abs/2208.09392)

## Other Material (Temporary)
* [Cold Diffusion Material](https://nimble-capri-8e2.notion.site/Cold-Diffusion-b3a6bdce9c2d4c0097aeb814bb86b2ea?pvs=4)

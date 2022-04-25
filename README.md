<h1>GANs based on Compact Vision Transformers</h1>

<h3> <u> OVERVIEW: </u> </h3>

![image](https://user-images.githubusercontent.com/47019139/165015471-1932b983-33e0-4696-ae9a-93eae9107276.png)

![image](https://user-images.githubusercontent.com/47019139/165015480-37f19579-1c6e-4683-92c6-a730054aadcf.png)

-------------------------------------------------------------------------------------------------------- 
<h3> ABOUT THE REPO: </h3>
<h3> 1. File structure </h3>
CONFIGS – This folder contains helper codes. <br><br>
MODELS – This folder contains different models which are required for training. <br><br>
OUTPUT/TRAIN – This folder contains results of CCT on CIFAR-10 and CIFAR-100. <br><br>
PLOTS – This folder contains plots for train/test loss and accuracy. <br><br>
SRC – This folder contains necessary codes for CVT, CCT and ViTs. <br><br>
UTILS – This folder contains various miscellanous code files. <br><br>
VITGAN.sbatch – sbatch file for running experiment for ViTGAN <br><br>
cifar10.sbatch – sbatch file for running CCT/CVT experiment on CIFAR-10 dataset. <br><br>
cifar100.sbatch – sbatch file for running CCT/CVT experiment on CIFAR-100 dataset. <br><br>
main.py – This python file contains the code being called in SLURM jobs. <br><br>
requirements.txt - This text file contains all the dependencies for the CONDA environment. <br><br>
train.py – This python file contains the code for training the model. <br><br>

-------------------------------------------------------------------------------------------------------- 
<h3> 2. How to clone </h3>
(Make sure you have Git Bash installed)
Run a Git Bash terminal in the folder you want to clone in and use the following command: <br>
git clone https://github.com/MohitK29/compact-transformers.git <br>
  
-------------------------------------------------------------------------------------------------------- 
<h3> 3. How to run the code </h3>

<h4> Train the model: How to train the model using SLURM jobs on HPC <br><br>
<li>Step 1: Create a conda environment <br>
Check out the following link to do so:  <br>
Managing environments — https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/packages/conda-environments <br><br>
<li>Step 2: For installing requirements
Run the following commands to set up the environment: <br><br>
pip install -r requirements.txt <br><br>
<li>Step 3: Run the following SLURM command by running the following command ‘bestmodel.sbatch’: <br>
sbatch bestmodel.sbatch <br><br>
This command will create a .out file and two .png files one for training/test loss and one for accuracy. It will run 200 epochs for the model and will print both losses and accuracies for each epoch. <br>
(Warning: This process takes approx. 45-60 mins for both training and testing combined). <br><br>
--------------------------------------------------------------------------------------------------------  



  <h2>REFERENCES:</h2>
<li>[1] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. ”Imagenet classification with deep convolutional
neural networks.” Advances in neural information processing systems 25 (2012).
<li>[2] Simonyan, Karen, and Andrew Zisserman. ”Very deep convolutional networks for large-scale image recognition.” arXiv preprint arXiv:1409.1556 (2014).
<li>[3] Szegedy, Christian, et al. ”Inception-v4, inception-resnet and the impact of residual connections on learning.” Thirty-first AAAI conference on artificial intelligence. 2017.
<li>[4] He, Kaiming, et al. ”Deep residual learning for image recognition.” Proceedings of the IEEE conference on
computer vision and pattern recognition. 2016.
<li>[5] Tan, Mingxing, and Quoc Le. ”Efficientnet: Rethinking model scaling for convolutional neural networks.”
International conference on machine learning. PMLR, 2019.
<li>[6] Zagoruyko, Sergey, and Nikos Komodakis. ”Wide residual networks.” arXiv preprint arXiv:1605.07146
(2016).
<li>[7] Zhang, Michael, et al. ”Lookahead optimizer: k steps forward, 1 step back.” Advances in Neural Information Processing Systems 32 (2019).
<li>[8] Krizhevsky, Alex, and Geoffrey Hinton. ”Learning multiple layers of features from tiny images.” (2009): 7.
<li>[9] Shorten, C., Khoshgoftaar, T.M. A survey on Image Data Augmentation for Deep Learning. J Big Data 6,
60 (2019). https://doi.org/10.1186/s40537-019-0197-0
<li>[10] Bergstra, James, and Yoshua Bengio. ”Random search for hyper-parameter optimization.” Journal of
machine learning research 13.2 (2012).
<li>[11] Yu, Tong, and Hong Zhu. ”Hyper-parameter optimization: A review of algorithms and applications.” arXiv
preprint arXiv:2003.05689 (2020).
<li>[12] https://github.com/kuangliu/pytorch-cifar

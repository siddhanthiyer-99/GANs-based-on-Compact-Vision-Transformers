<h1>Implementations of ResNet architecture on CIFAR10 with less than 5M Parameters – Deep Learning Mini Project 1</h1>

<h3> <u> OUR ARCHITECTURE: </u> </h3>

![architecture](https://user-images.githubusercontent.com/47019139/160049019-05f6d487-c9c3-4116-b435-36c512c0a7b1.PNG)

Here,
<ul>
  
<li>N: # Residual Layers = 3
<li>Bi: # Residual blocks in Residual Layer i = 2
<li>Ci: # channels in Residual Layer i = 64
<li>Fi: Conv. kernel size in Residual Layer i = 3 x 3
<li>Ki: Skip connection kernel size in Residual Layer i = 1 x 1
<li>P: Average pool kernel size = 8 x 8
</ul>

-------------------------------------------------------------------------------------------------------- 
<h3> ABOUT THE REPO: </h3>
<h3> 1. File structure </h3>
OUTPUTS – This folder contains all the outputs in ‘.out’ format, of the different experiments with different parameters. To view contents of the folder, use the “cat” command. <br><br>
PLOTS – This folder contains all the different graphs plotted for each corresponding experiment. They contain .png files and can be opened on Github itself. <br><br>
SBATCH – This folder contains all the different ‘.sbatch’ files created for each corresponding experiment. They are used to assign Slurm jobs, use the command “sbatch filename.sbatch” to run the particular experiment. <br><br>
best_model.out – This is the output file generated for our model which produced the best results. <br><br>
best_model_acc.png & best_model_loss.png – These are the train/test accuracy graph and loss graph for our model. 
bestmodel.sbatch – The sbatch for this model is bestmodel.sbatch. <br><br>
main.py – Python file being ran by the slurm command which contains our training logic and saves the best model weights in project1_model.pt file. <br><br>
project1_model.pt – This is a PyTorch file for our best architecture with saved parameters that can be loaded for testing. <br><br>
requirements.txt – This file contains all the different libraries used for this project. <br><br>
test.py – Python program to run the model saved in project1_model.pt on CIFAR10 testset. <br><br>
utils.py – Python program which is being used by ‘main.py’ to import different functionalities such as Progress bar and computing the mean and standard deviation value of dataset. <br><br>

-------------------------------------------------------------------------------------------------------- 
<h3> 2. How to clone </h3>
(Make sure you have Git Bash installed)
Run a Git Bash terminal in the folder you want to clone in and use the following command: <br>
git clone https://github.com/dhyani15/resnet-implementation.git <br>
  
-------------------------------------------------------------------------------------------------------- 
<h3> 3. How to test the code </h3>
<h4> Method 1(To only test the trained model on cifar10 testset): Using ‘test.py’ (Saved parameters and weights) - Recommended </h4>
(Make sure the file ‘project1_model.pt’ and ‘test.py’ are in the same folder) <br><br>
<li>Step 1: Create a conda environment <br>
Check out the following link to do so:  <br>
Managing environments — https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/packages/conda-environments <br><br>
<li>Step 2: For installing requirements
Run the following commands to set up the environment: <br><br>
pip install -r requirements.txt <br><br>
<li>Step 3. Run ‘test.py’ to display accuracy
Run the python command: <br>
python test.py <br><br>
This program uses the saved model ‘project1_model.pt’ and displays the accuracy as a Tensor. <br><br>
If you are planning to test our model using your own test script, make sure it has the following command<br>
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')<br>
model = project1_model().to(device)<br>
model_path = './project1_model.pt'<br>
checkpoint = torch.load(model_path, map_location=device)<br>
model = torch.nn.DataParallel(model)<br>
model.load_state_dict(checkpoint, strict=False)<br>
--------------------------------------------------------------------------------------------------------  
  <h4> Method 2(to train our model from scratch) : How to retrain the model using SLURM jobs on HPC </h4>
<li>Make sure you have cloned this repo on your hpc and repeat step 1 & step 2 from Method 1 <br>
<li>Step 3: Run the following SLURM command by running the following command ‘bestmodel.sbatch’: <br>
sbatch bestmodel.sbatch <br><br>
This command will create a .out file and two .png files one for training/test loss and one for accuracy. It will run 200 epochs for the model and will print both losses and accuracies for each epoch. <br>
(Warning: This process takes approx. 45-60 mins for both training and testing combined). <br><br>

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

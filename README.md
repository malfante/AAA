# Automatic Analysis Architecture 

Welcome to this automatic classification scheme! Please carefully read the following before asking questions :)

If used, the software should be credited as follow:    

**Automatic Analysis Architecture, M. MALFANTE, J. MARS, M. DALLA MURA**   
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1216028.svg)](https://doi.org/10.5281/zenodo.1216028)   

and the original paper for which the code was developped should be cited:   

**Malfante, M., Dalla Mura, M., Metaxian, J. P., Mars, J. I., Macedo, O., & Inza, A. (2018). Machine Learning for Volcano-Seismic Signals: Challenges and Perspectives. IEEE Signal Processing Magazine, 35(2), 20-30.**  

We thank you for the respect of the authors work.


## What is this code for?

This code answers three different purposes:

1. The automatic classification of continuous signals, stored in recording files (.wav, .sac, etc)
2. The automatic classification of discrete (sparse) events, stored as numpy.array objects. 
3. The automatic classification of discrete (sparse) events, with real time conditions and data requests. 

The roadmap to help you chose a usecase is simple:  

- If your data are continuous recordings in which you want to detect and classify certain classes of event, go for Usecase 1. 
- If your data are already detected events that you want to classify, you have one more question to answer. 
	- Are you data stored in recordings ? If so, read and shape your data in numpy.array format and go for Usecase 2. 
	- If you are dealing with real time data request, then go for Usecase 3. 

## Set up and requierements needed to run the code
This code was developed under Python 3, and needs the following libraries. Those libraries need to be previously installed.

- `numpy==1.13.3`
- `scipy==0.18.1`
- `pandas==0.19.2`
- `matplotlib==1.5.3`
- `numpy`
- `obspy==1.0.2`
- `python_speech_features==0.4`
- `sympy==1.0`
- `soundfile==0.8.1`
- `scikit-learn==0.18.1`

To install the correct version of python, along with the library, you can use miniconda environment manager. 

1- Download and install miniconda: `https://conda.io/miniconda.html` 
NB: By installing miniconda, your `.bachrc` will be modified with the following line : 

> 	added by Anaconda3 4.3.0 installer  
>	export PATH="/home/user/anaconda3/bin:$PATH"

We suggest you replace them by 
>	"$PATH:/home/user/anaconda3/bin"

It will leave your computer configuration unchanged (in particular, your previous versions of python will still be used)

2- Create and activate your working environment (in a terminal session):  
`conda create -n AAA python=3.6`  
`source activate myEnvName`

3- Install the libraries:  
`pip install --upgrade pip`  
`pip install -r AAA_requierements.txt`. 

4- Run the code (see next section)

5- Quit the working environment:  
`source deactivate` 



## How to run the code?

Each usecase can be run from two different ways, depending on your preferences and what you intend to do with this code.

##### Option 1
Either using a bash script and path to settings files as input arguments. Bash scripts will run the appropriate Python scripts and properly save and display results. This method is 'the official one'. In your favorite terminal window, start by moving to the `automatic_processing` folder using `cd` command. Then run one of the usecase makefile as follow:

		bash make_usecase1.sh setting_file action verbatim
	or 
		
		bash make_usecase2.sh setting_file verbatim
	or
		
		bash make_usecase3.sh setting_file action verbatim
		
setting\_file are stored in the configuration folder and contain all settings needed to run an analysis. Depending on the usecase you have choosen, the information requested in setting_files can change. Please refer yourself to the setting\_file sections for more details.	
		
##### Option 2
Either by using Python playgrounds scripts, if you need a playground where to experiment. Those script are easily found, they all are called something like `PLAYGROUND_USECASE1.py` or similarly. Path to settings and input arguments are 'hard coded' at the beginning of each playground file. To run on of those scripts, simply go to the `automatic_processing` folder and run one on the following commands: 

		python3 PLAYGROUND_something_something.py 
		

Both configurations are basically going torward the same analysis but you might prefer one or the other depending on what you want to do. 


##### More detail on the input arguments 

- `setting_file`: path to the setting file. Traditionnaly, setting file are stored in the `config` folder. The next section gives details on the formatting of configuration files. 

- `verbatim`: 
	- 0 - quiet
	- 1 - some information regarding general steps
	- 2 - more detailed information
	- 3 - all details

	
- `action`: 
	- training to train and save a model 
	- analyzing to run the analysis 
	- make_decision to make decision from the output probabilities output from the analysis
	- display if it is of interest to you.  
	
Obviously, the various actions should be run in the order ... analysis cannot be run withour a trained model. 

When running the code for a new application, a specific folder is created for all results. 


## Configuration files 
	 	
All the settings related to a new project or a new run are indicated in a setting main setting. 

It contains information regarding the project paths, the considered application, the signals preprocessing, the features used (linked to a dedicated feature configuration file), and the learning algorithms. 

Extra information regarding the wanted analysis, the data to analyze and display parameters are indicated in a separate configuration file, which format depends on the usecase. 

So, for each configuration, 3 configuration files are considered:

- the general setting file, contained in `config/general` folder
- the feature setting file, contained in `config/specific/features` folder
- the configuration file specific to the wanted analysis, contained in `config/specific/usecaseXX/`

Commented examples for each of the setting files are available (but keep in mind that json files do not support comments, so those files are simply there as examples.)


## More info	

If you still have questions, try running and exploring the code. 
The playground files are relatively easy to play with. 

If you still have question, fell free to ask ! 

**Contact:** marielle.malfante@gmail.com


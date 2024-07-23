# MuscleSynergyExtractionBench
Codebase to investigate Muscle Synergy Extraction techniques

Codes for testing autoencoder for synergy extraction on upper limb multi-plane and multi-directional movements in comparison with the NMF are provided. 
Two main python scripts are provided: 
- main.py: extracts synergies with the autoencoder trained on all the planes
- main_sp.py: extracts synergies with the autoencoder trained on a single plane

Both the scripts take as input a .mat file with training and test data. Scripts and functions that are used by the main codes are reported in the utils.py file. The code is exactly the one used for obtaining the results of the paper (“On autoencoders for extracting muscle synergies: a study in highly variable upper limb movements”) tested on example synthetic data (S00_input.mat). Therefore, when using these codes with other datasets, data should be correctly formatted and code should be adapted accordingly. 

Since the dataset used in the paper cannot be publicly shared, an example dataset and the script used to generate it are provided. The matlab script (generate_synthetic_data.m) generates and saves simulated biomimetic data of biphasic muscle activations of 16 muscle in 9 directions of 5 planes. Repetitions of movement are obtained adding noise. Nine repetitions for each plane are used as training data and one repetition is used for testing. The simulated dataset has the same dimension and characteristics of the real dataset used in the paper. The simulated data are based on biomimetic principles but they do not correspond to the real activations. The code can be used for generating data to test the autoencoder that give results similar to the ones obtained in the paper. Since the code contains random factors, every time the code is run, the simulated datasers are different.  

S00_input.mat contains simulated data resembling the data used in the paper. These data contain matrices of 16 muscles and 9 directions of movement that are subdivided in 5 planes of movement (f: frontal, h: horizontal, l: left, r: right, u: up). Nine movement repetitions are used for training and one repetition is used for testing. The dimensions are: 9x900x16 for training data, 1x900x16 for testing data.

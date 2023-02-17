# Installation and Usage

The source code is written in Python 3 and tested using Python 3.9 on Mac OS. It is recommended to use a virtual Python environment for this setup.

Note: This might not work with Apple silicon chips due to underlying library changes

## Installation and Environment Setup
Follow the instructions below to setup environment and run the source code for reproduction.

Follow these steps to create a virtual environment:

1. Install Anaconda [[installation guide](https://docs.anaconda.com/anaconda/install/)].

2. Create a new Python environment. Run on command line:
```
conda create --name ensemblefairness python=3.9
conda activate ensemblefairness
```
The shell should look like: `(ensemblefairness) $ `. Now, continue to step 4 and install packages using this shell.
When you are done with the experiments, you can exit this virtual environment by `conda deactivate`.

3. Clone/Download this repository. This will clone both data and code to run the benchmark.

4. Navigate to the repository and install required packages:
```
pip3 install -r Requirements.txt
```


## Run Experiments

To get the fairness and accuracy report for any model, run the evaluate.py python script by passing the .pkl file for the corresponding model which can be found in the results folder for all datasets and ensemble models.

E.g. to generate fairness and accuracy report for the Gradient Boosting Model "6-titanic-best-working-classifier.pkl" in the Titanic Dataset, run the following command:

python3 evaluate.py "Titanic/Results/GBC/6-titanic-best-working-classifier.pkl"
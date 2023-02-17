Due to size of the dataset, the complete artifact with the dataset can be download from Zenodo (https://zenodo.org/record/7647900#.Y-70eOzMJjV)

# Fair Ensembles
The repo contains the benchmark models, datasets and results for the following paper submission at ICSE 2023.

**Title**: Towards Understanding Fairness and its Composition in Ensemble Machine Learning

**Abstract** Machine Learning (ML) software has been widely adopted in modern society, with reported fairness implications for minority groups based on race, sex, age, etc. Many recent works have proposed methods to measure and mitigate algorithmic bias in ML models. The existing approaches focus on single classifier-based ML models. However, real-world ML models are often composed of multiple independent or dependent learners in an ensemble (e.g., Random Forest), where the fairness composes in a non-trivial way. \textit{How does fairness compose in ensembles? What are the fairness impacts of the learners on the ultimate fairness of the ensemble?} Furthermore, studies have shown that hyperparameters influence the fairness of ML models. Ensemble hyperparameters are more complex since they affect how learners are combined in different categories of ensembles. Understanding the impact of ensemble hyperparameters on fairness will help programmers in designing fair ensembles. Today, we do not understand these fully for different ensemble algorithms. In this paper, we comprehensively study popular real-world ensembles: bagging, boosting, stacking and voting. We have developed a benchmark of 168 ensemble models collected from Kaggle on four popular fairness datasets. We use existing fairness metrics to understand the composition of fairness. Our results show that ensembles can be designed to be fairer without using mitigation techniques. We also identify the interplay between fairness composition and data characteristics to guide fair ensemble design. Finally, our benchmark can be leveraged for further research on fair ensembles. To the best of our knowledge, this is one of the first and largest studies on fairness composition in ensembles yet presented in the literature.

To get the fairness and accuracy report for any model, run the evaluate.py python script by passing the .pkl file for the corresponding model which can be found in the results folder for all datasets and ensemble models.

E.g. to generate fairness and accuracy report for the Gradient Boosting Model "6-titanic-best-working-classifier.pkl" in the Titanic Dataset, run the following command:

python3 evaluate.py "Titanic/Results/GBC/6-titanic-best-working-classifier.pkl"


## Index

> 1. Datasets <br>
	- [Adult Census](https://gitlab.com/anonymousdot/fair-ensemble/-/tree/main/AdultNoteBook/Data) <br>
	- [Bank Marketing](BankMarketingNoteBook/Data) <br>
	- [Titanic](Titanic/Data) <br>
	- [German Credit](GermanCredit/Data) <br>

> 2. Benchmark
	- All the ensemble models used in the study and their scripts are shared in the following directories: <br>
		- [Adult Census](https://gitlab.com/anonymousdot/fair-ensemble/-/tree/main/AdultNoteBook/Kernels) <br>
		- [Bank Marketing](BankMarketingNoteBook/Kernels) <br>
		- [Titanic](Titanic/Kernels) <br>
		- [German Credit](GermanCredit/Kernels) <br>

> 3. Results
	- For reproduction and validation of results, we provide the exact train/test splits, saved model (.pkl) and measures in a .csv. Note: Due to randomization in some models, output might vary a bit. We take mean of 10 runs to reduce the variance <br>
		- [Adult Census](https://gitlab.com/anonymousdot/fair-ensemble/-/tree/main/AdultNoteBook/Results) <br>
		- [Bank Marketing](BankMarketingNoteBook/Results) <br>
		- [Titanic](Titanic/Results) <br>
		- [German Credit](GermanCredit/Results) <br>


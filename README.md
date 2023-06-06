# Internship

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Requirements](#requirements)
* [Used models](#used-models)
* [How to run](#how-to-run)

## General info
This project basically cluseters data depending on the tokens it generates from preprocessing using nltk library. WordtoVec model is used for word embedding and then different clustering algo to find suitable cluster
	
## Technologies
The whole project is done on [transac-nar-new.ipynb](https://github.com/Asif-droid/Internship/blob/main/transac-nar-new.ipynb)
## Requirements
The required technologies to run this project is included here  (https://github.com/Asif-droid/Internship/blob/main/requirements.txt)
# Used models
* Kmeans
* Minibatch_Kmeans
* Bisecting_Kmeans
* Dbscan
* Hierarchy clustering


## How to run
* Download or clone the repo 
* open in local machine
* meet the requirements run-
	pip install -r requirements.txt
* Open the test_script file.
* Give locations of dataset and trained model 
* Can adjust the values for Hierarchy clustering and Dbscan (defult is mx_d=1.5 for Hierarchy and eps=.55, min_samples=1 for dbscan)
* Run the file
* For more clearificaion see  [transac-nar-new.ipynb](https://github.com/Asif-droid/Internship/blob/main/transac-nar-new.ipynb)





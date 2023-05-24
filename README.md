# MLOps

This repo documents my understanding of MLOps. The structure of my notes are as follows:

# Table of content

1. [Introduction](#1)
    1. [What is MLOps](#2)
    2. [Environment preparation](#3)
    3. [Issues with jupyter notebooks and why we need MLOps (experiment tracker, model registry, ML pipeline and some best practices)](#4)
    4. [MLOps maturity model](#5)

2. [Experiment tracking and model management](#6)
    1. [Introduction to experiment tracking and MLflow](#7)
        1. [Some terminologies](#8)
        2. [What’s experiment tracking?](#9)
        3. [Why is experiment tracking so important?](#10)
        4. [MLflow](#11)
        5. [Tracking experiments with MLflow](#12)
        
8. [Prerequisites (deployment and Docker)](#12)
9. [References](#13)








<a name="1"></a>
## 1. Introduction

<a name="2"></b>
### What is MLOps

+ MLOps is a set of best practises to put the machine learning model to production. 
+ The simplified version of an ML project is depicted in the following:

![](https://github.com/DanialArab/images/blob/main/MLOPS/ML%20project%20steps.png?raw=true)

In this course we will focus on the second (train) and third step (operate). 

<a name="3"></b>
### Environment preparation

I set up my environemnt in my VM Ubuntu. I need to install:
+ Anaconda, for that:

            wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
            bash Anaconda3-2023.03-1-Linux-x86_64.sh
        
+ Docker

            sudo apt install docker.io
            
+ run docker without sudo

            sudo usermod -aG docker $USER (https://docs.docker.com/engine/install/linux-postinstall/)
        
+ Docker compose 

Docker Compose is a tool for running multi-container applications on Docker defined using the Compose file format. A Compose file is used to define how one or more containers that make up your application are configured. Once you have a Compose file, you can create and start your application with a single command: docker compose up.

            mkdir soft
            cd soft
            wget https://github.com/docker/compose/releases/download/v2.18.0/docker-compose-linux-x86_64
            mv docker-compose-linux-x86_64 docker-compose
            chmod +x docker-compose (to make it executable)
            nano .bashrc (to be able to access the soft folder, which contains the docker-compose, from anywhere I need to modify my PATH variable:)
                export PATH="${HOME}/soft:${PATH}"
            source .bashrc

+ Clone the course repo 

            git clone https://github.com/DataTalksClub/mlops-zoomcamp.git
            cd mlops-zoomcamp/
            
+ I also need to install pyarrow to be able to read parquet data (NY Taxi data was switched from csv to parquet):

            pip install pyarrow

<a name="4"></a>
### Issues with jupyter notebooks and why we need MLOps (experiment tracker, model registry, ML pipeline and some best practices)

Notebooks are usually intended for experimentation and beyond this experimentation they have the following drawbacks:

+ hard to remember the order by which the cells need to be executed
+ hard to remember which cells we really need 
+ if we train different models with different parameters when experimenting, when later we come back to the notebook we lost all the history and the model's performances vs. various parameters unless we took a record of the details like in a spreedsheet documenting some metrics of the model performance along with the parameters tried, which is good but not ideal. The ideal is to log all the metrics to a special place called **experiment tracker**, where we can always go back and see all the preserved history. For experiment tracking we use a tool called **MLflow.**
+ in a notebook we may save a model through pickle, when coming back we may not exactly know which model was saved, so that is why we save the model in a place called **model registry** where we keep all the models along with the metrics in the experiment tarcker, which is great for future reference with no ambiguity.
+ we need to decompose our notebook and trun it into something that can be easily re-executed, which is called **ML pipeline**, like if we want to re-train the model what are the cells that we need to re-execute! We can parameterize our ML pipeline like in the future we just need to tune these parameters to easily re-execute the training through running a python script containing the ML pipeline. We use tools and best practices to achieve this, like using **Prefect** and **Kubeflow.** 

So it is recommended to put the codes in a python script to turn the notebook into a more modular format. 

<a name="5"></a>
### MLOps maturity model

good reference from microsoft: https://learn.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model

The purpose of the maturity model is to help clarify the Machine Learning Operations (MLOps) principles and practices. The maturity model shows the continuous improvement in the creation and operation of a production level machine learning application environment. You can use it as a metric for establishing the progressive requirements needed to measure the maturity of a machine learning production environment and its associated processes.

The MLOps maturity model encompasses five levels of technical capability:

+ No MLOps
+ DevOps but no MLOps
+ Automated Training
+ Automated Model Deployment
+ Full MLOps Automated Operations


<a name="6"></a>
## 2. Experiment tracking and model management

<a name="7"></a>
### Introduction to experiment tracking and MLflow

<a name="8"></a>
#### Some terminologies

+ ML experiment: the process of building an ML Model
+ Experiment run: each trial in an ML experiment 
+ Run artifact: any file that is associated with an ML run
+ Experiment metadata: all the info related to the ML experiment like the source
code used, the name of the user, etc.

<a name="9"></a>
#### What’s experiment tracking?

Experiment tracking is the process of keeping track of all the relevant information from an ML experiment, which includes:

+ Source code
+ Environment
+ Data
+ Model
+ Hyperparameters
+ Metrics
+ ...

The above info entities which we keep track of as an experiment tracking depends on the problem in hand like if an ML engineer or Data Scientist tries to play with hyperparameters he/she wants to keep track of those hyperparameters along with the the values tried, or like if he/she tried different processes on the data before feeding into the model these processes' details should be kept track of. So all in all the main point here is that the info entities we keep track of as an experiment tracking is not unique and it all depends on the problem in hand, but the above list is somehow general/standard guidlines. 

<a name="10"></a>
#### Why is experiment tracking so important?

In general, because of these 3 main reasons:

+ Reproducibility: as a scientist we want to reproduce the results
+ Organization
+ Optimization

<a name="11"></a>
#### MLflow

MLflow is a tool we use to perform experiment tracking. **MLflow is an open source platform for the machine learning lifecycle.** The ML lifecycle refers to the whole process of building and maintaining of an ML project. 

On the contrary with Kubeflow, which requires certain infra on top of the library to work, MLflow is just a Python package that can be installed with pip, and it contains four main modules:

+ Tracking
+ Models
+ Model Registry
+ Projects (out of scope for this course, take a look on your own)

<a name="12"></a>
#### Tracking experiments with MLflow

The MLflow Tracking module allows you to organize your experiments into runs, and **manually** to keep track of:

+ Parameters
+ Metrics
+ Metadata
+ Artifacts
+ Models

some explanations:

-- for example the path to the training dataset could be one parameter, since in the future if you want to change the dataset you can just easily change that parameter. Or if you apply various preprocessing to the data you can add those preprocessing as a parameter. 

-- You may have tags like the name of developer etc. which could be kept track of as metadata.

-- If you train an ML model and then performed some visualizations of the data, this can be considered as artifact. Also you can look at the dataset as an artifact but of course this does not scale very well because your data will be duplicated and you can come up with a better solution for that. 


Along with the above information that you can keep track of, MLflow **automatically** logs extra information about the run:

+ Source code
+ Version of the code (git commit)
+ Start and end time
+ Author

<a name="13"></a>
### Getting started with MLflow

Let's first create a separate conda environemnt and install all the packages and libraries: 

        conda create -n exp-tracking-env python=3.10.9
        [conda info --envs](http://127.0.0.1:5000 ) http://127.0.0.1:5000 
        conda activate exp-tracking-env
        pip install -r requirements.txt
        pip list

requirements.txt includes

        mlflow
        jupyter
        scikit-learn
        pandas
        seaborn
        hyperopt
        xgboost
        fastparquet
        boto3
        
Some quick note:

+ hyperopt:
Hyperopt is a Python library for hyperparameter optimization, which is the process of finding the best set of hyperparameters for a machine learning model. It provides a flexible and efficient framework for defining and searching over a hyperparameter search space.

+ fastparquet:
Fastparquet is a Python library for reading and writing Parquet files efficiently. Parquet is a columnar storage file format that is highly optimized for analytical processing, particularly for big data workloads. Fastparquet is designed to provide fast and memory-efficient I/O operations for working with Parquet files.

to get access to the mlflow ui:

        mlflow ui --backend-store-uri sqlite:///mlflow.db

then in my code, I add:

        import mlflow 

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment('nyc-taxi-experiment')

also:

        with mlflow.start_run():

            mlflow.set_tag("developer", 'danial')

            mlflow.log_param('train-data-path', parent_directory + 'green_tripdata_2021-01.parquet')
            mlflow.log_param('valid-data-path', parent_directory + 'green_tripdata_2021-02.parquet')

            alpha = 0.1
            mlflow.log_param('alpha', alpha)

            lr = Lasso(alpha)
            lr.fit(X_train, y_train)

            y_pred = lr.predict(X_val)

            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric('rmse', rmse)

<a name="8"></a>
## 8. Prerequisites (deployment and Docker)

https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/05-deployment

![](https://raw.githubusercontent.com/DanialArab/images/main/MLOPS/Deployment.PNG)

<a name="9"></a>
## 9. References

https://github.com/DataTalksClub/mlops-zoomcamp


my progress: up to 2.3 Experiment tracking with MLflow

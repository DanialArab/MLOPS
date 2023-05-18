# MLOps

This repo documents my understanding of MLOps. The structure of my notes are as follows:

# Table of content

1. [Introduction](#1)
    1. [What is MLOps](#2)
    2. [Environment preparation](#3)
    3. [Issues with jupyter notebooks and why we need MLOps (experiment tracker, model registry, ML pipeline and some best practices)](#4)
    4. [MLOps maturity model](#5)

2. [Experiment tracking and model management](#6)


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

            I already installed docker engine on my machine when I took Mosh's docker course
        
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

<a name="8"></a>
## 8. Prerequisites (deployment and Docker)

https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/05-deployment

![](https://raw.githubusercontent.com/DanialArab/images/main/MLOPS/Deployment.PNG)

<a name="9"></a>
## 9. References

https://github.com/DataTalksClub/mlops-zoomcamp


my progress: up to homework of the first module

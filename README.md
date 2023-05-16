# MLOps

This repo documents my understanding of MLOps. The structure of my notes are as follows:

# Table of content

1. [Introduction](#1)
    1. [What is MLOps](#2)
    2. [Environment preparation](#3)
    3. [MLOps maturity model](#4)
    4. [Why do we need MLOps](#5)

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

<a name="6"></a>
## 3. Experiment tracking and model management

<a name="8"></a>
## 8. Prerequisites (deployment and Docker)

https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/05-deployment

![](https://raw.githubusercontent.com/DanialArab/images/main/MLOPS/Deployment.PNG)

<a name="9"></a>
## 9. References

https://github.com/DataTalksClub/mlops-zoomcamp

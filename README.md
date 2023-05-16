# MLOps

This repo documents my understanding of MLOps. The structure of my notes are as follows:

# Table of content

1. [Introduction](#1)
    1. [What is MLOPS](#2)
    2. [MLOps maturity model](#3)
    3. [Why do we need MLOps](#4)

2. [Environment preparation](#5)
2. [Experiment tracking and model management](#6)


8. [Prerequisites (deployment and Docker)](#12)
9. [References](#13)


<a name="1"></a>
## 1. Introduction

+ MLOps is a set of best practises to put the machine learning model to production. 
+ The simplified version of an ML project is depicted in the following:

![](https://github.com/DanialArab/images/blob/main/MLOPS/ML%20project%20steps.png?raw=true)

In this course we will focus on the second (train) and third step (operate). 

<a name="5"></a>
## 2. Environment preparation

I set up my environemnt in my VM Ubuntu. I need to install:
+ Anaconda, for that:
        wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
        bash Anaconda3-2023.03-1-Linux-x86_64.sh
        
+ Docker
        I already installed docker engine on my machine when I took Mosh's docker course
        
+ Docker compose 
        


<a name="6"></a>
## 3. Experiment tracking and model management

<a name="8"></a>
## 8. Prerequisites (deployment and Docker)

https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/05-deployment

![](https://raw.githubusercontent.com/DanialArab/images/main/MLOPS/Deployment.PNG)

<a name="9"></a>
## 9. References

https://github.com/DataTalksClub/mlops-zoomcamp

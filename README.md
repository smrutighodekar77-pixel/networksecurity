Network Security / Phishing Detection – End-to-End ML Deployment 🚀

An end-to-end Machine Learning + MLOps project that detects whether a website is Phishing (Malicious) or Legitimate based on extracted URL and domain features.

This project covers the complete ML lifecycle — from data ingestion and training to production-grade deployment on AWS with CI/CD.

🔍 Problem Statement

Phishing websites pose serious security risks by imitating trusted sources to steal sensitive information.

The goal of this project is to:

Analyze URL-based features

Train a machine learning model

Deploy the model as a scalable web service

Enable batch predictions via CSV upload

🧠 Solution Overview

The system accepts a CSV file containing URL features and returns predictions indicating whether each website is phishing (1) or legitimate (0).

The application is deployed using FastAPI, containerized with Docker, and continuously deployed using GitHub Actions on AWS EC2 with images stored in Amazon ECR.

🛠️ Tech Stack

Programming & ML

Python

Scikit-learn

Pandas, NumPy

Backend

FastAPI

Uvicorn

MLOps & DevOps

Docker

GitHub Actions (CI/CD)

AWS EC2

AWS ECR


⚙️ Features

End-to-end ML pipeline implementation

CSV-based batch predictions

Interactive Swagger UI

Dockerized application

Automated CI/CD pipeline

Cloud deployment on AWS

Production-ready architecture


📥 Input Format (CSV)

The input CSV must contain the following features:

having_IP_Address,URL_Length,Shortining_Service,having_At_Symbol,
double_slash_redirecting,Prefix_Suffix,having_Sub_Domain,
SSLfinal_State,Domain_registeration_length,Favicon,port,
HTTPS_token,Request_URL,URL_of_Anchor,Links_in_tags,SFH,
Submitting_to_email,Abnormal_URL,Redirect,on_mouseover,
RightClick,popUpWidnow,Iframe,age_of_domain,DNSRecord,
web_traffic,Page_Rank,Google_Index,Links_pointing_to_page,
Statistical_report


Each row represents one website.


🔄 CI/CD Pipeline

Implemented using GitHub Actions:

Continuous Integration

Code checkout

Tests & lint checks

Continuous Delivery

Docker image build

Push image to Amazon ECR

Continuous Deployment

Pull latest image on EC2

Stop old container

Run updated container


☁️ AWS Deployment

EC2 → Application hosting

ECR → Docker image registry

Self-hosted GitHub Runner → Automated deployments


📌 Learning Outcomes

Built production-ready ML pipelines

Learned real-world MLOps practices

Understood CI/CD automation

Hands-on AWS deployment experience

Bridged gap between ML and DevOps


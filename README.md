# MLOps Assignment 1 - Boston Housing Price Prediction

This repository implements and automates a complete machine learning workflow to **predict house prices** using classical machine learning models.

The dataset used is the **Boston Housing dataset** (fetched from the CMU server, since `load_boston` is deprecated in scikit-learn).

We implement and evaluate two regression models:
- **Decision Tree Regressor** (`train.py`)
- **Kernel Ridge Regressor** (`train2.py`)

---

## Repository Structure
MLOps_A1/
├── misc.py # Helper functions (data loading, preprocessing, train/eval)
├── train.py # DecisionTreeRegressor training + evaluation
├── train2.py # KernelRidge training + evaluation
├── requirements.txt # Python dependencies
├── README.md # Instructions to install and run
└── .github/
└── workflows/
└── ci.yml # GitHub Actions workflow


Report Section: Git and Branch Workflow
1. Initialize Git

In the project folder:

cd C:\Users\ayush\MLOps_A1
git init

2. Configure Git (one-time setup)
git config --global user.name "Ayush Chauhan"
git config --global user.email "yourgithubemail@example.com"


Verify:

git config --list

3. Stage and Commit Files
git add README.md misc.py train.py train2.py requirements.txt .github/workflows/ci.yml
git commit -m "Initial commit - add scripts and requirements"

4. Rename Branch
git branch -M main

5. Link Local Repo to GitHub

Created empty GitHub repo MLOps_A1 and linked it:

git remote add origin https://github.com/anonymous666666666/MLOps_A1.git

6. Push Code to GitHub
git push -u origin main


Used GitHub username and PAT (Personal Access Token) for authentication.

7. Create dtree Branch
git checkout -b dtree
git add .
git commit -m "Add DecisionTree training script"
git push -u origin dtree

8. Merge dtree into main
git checkout main
git merge dtree --no-ff -m "Merge dtree into main"
git push origin main

9. Create kernelridge Branch
git checkout -b kernelridge
git add .
git commit -m "Add KernelRidge model and CI workflow"
git push -u origin kernelridge

10. Verify GitHub Actions

On GitHub repo → Actions tab.

Workflow triggered on push to kernelridge.

Logs show training results from both:

train.py (DecisionTreeRegressor)

train2.py (KernelRidge)
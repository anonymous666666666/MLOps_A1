MLOps Assignment 1 - Boston Housing Price Prediction
This repository implements and automates a complete machine learning workflow to predict house
prices using classical machine learning models.
The dataset used is the Boston Housing dataset (fetched from the CMU server, since load_boston is
deprecated in scikit-learn).
We implement and evaluate two regression models:
- Decision Tree Regressor (train.py)
- Kernel Ridge Regressor (train2.py)
------------------------------------------------------------
Repository Structure
MLOps_A1/
|-- misc.py # Helper functions (data loading, preprocessing, train/eval)
|-- train.py # DecisionTreeRegressor training + evaluation
|-- train2.py # KernelRidge training + evaluation
|-- requirements.txt # Python dependencies
|-- README.md # Instructions to install and run
|-- .github/
 |-- workflows/
 |-- ci.yml # GitHub Actions workflow
------------------------------------------------------------
Installation Instructions
1. Clone the Repository
git clone https://github.com/anonymous666666666/MLOps_A1.git
cd MLOps_A1
2. Create and Activate Virtual Environment
Windows:
python -m venv venv
venv\Scripts\activate
macOS/Linux:
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
------------------------------------------------------------
Running the Code Locally
Run Decision Tree Regressor
python train.py
Run Kernel Ridge Regressor
python train2.py
------------------------------------------------------------
Git and Branch Workflow
1. Initialize Git
cd C:\Users\ayush\MLOps_A1
git init
2. Configure Git (one-time setup)
git config --global user.name "anonymous666666666"
git config --global user.email "aychauh@amazon.com"
git config --list
3. Stage and Commit Files
git add README.md misc.py train.py train2.py requirements.txt .github/workflows/ci.yml
git commit -m "Initial commit - add scripts and requirements"
4. Rename Branch
git branch -M main
5. Link Local Repo to GitHub
git remote add origin https://github.com/anonymous666666666/MLOps_A1.git
6. Push Code to GitHub
git push -u origin main
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
On GitHub repo -> Actions tab
Workflow triggered on push to kernelridge
Logs show training results from both train.py and train2.py
------------------------------------------------------------
Performance Comparison
Model | Test MSE | Test RMSE
------------------------|----------|-----------
Decision Tree Regressor | 10.4161 | 3.2274
Kernel Ridge Regressor | 476.2581 | 21.8233
------------------------------------------------------------
Author
- Name: Ayush Chauhan
- Roll Number: G24AI2059
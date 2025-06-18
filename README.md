# Machine Model And Implementation    

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : TELLAGORLA BALAMOHAN

*INTERN ID* : CT6WTDG727

*DOMAIN* : PYTHON PROGRAMMING

*DURATION* : 6 WEEKS

*MENTOR* : NEELA SANTOSH

# OVERVIEW OF PROJECT
This project implements a Machine Learning classification model using the Random Forest algorithm to predict whether a student will pass or fail based on their demographic and academic features.

It uses a structured dataset of student information to train the model and includes data preprocessing, feature engineering, training, testing, and evaluation—all handled using Python’s data science stack.

# TECHNOLOGIES USED

| Tool / Library           | Purpose                                                |
| ------------------------ | ------------------------------------------------------ |
| `pandas`                 | Data manipulation and analysis                         |
| `numpy`                  | Numerical operations                                   |
| `matplotlib` & `seaborn` | Data visualization                                     |
| `scikit-learn`           | Machine learning (modeling, preprocessing, evaluation) |

# 🌟 Features Implemented
1.Data loading and encoding of categorical features

2.Feature scaling using StandardScaler

3.Train-Test split using train_test_split

4.Model training using RandomForestClassifier

5.Model evaluation using:

   1.Accuracy score

   2.Classification report

   3.Confusion matrix

6.Visualization:

   1.Countplot of Passed vs Not Passed

   2.Confusion Matrix


# ▶️ How to Run This Project
1.Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn

2.Download the dataset:
Save this CSV file as student_data.csv in your project folder.

3.Run the script:

In VS Code, open the terminal and run:

  python main.py
  
4.Output:

-Console will display accuracy and classification report.

-A window will show visualizations for:

-Student Pass/Fail distribution

-Confusion Matrix of model performance.

# ⚙️ How It Works
1.Data Preprocessing: All non-numeric features are encoded using LabelEncoder, and then scaled.

2.Model Training: A Random Forest Classifier is trained on 80% of the data.

3.Prediction: The model makes predictions on unseen 20% test data.

4.Evaluation: Results are compared with actual labels using accuracy, precision, recall, and confusion matrix.











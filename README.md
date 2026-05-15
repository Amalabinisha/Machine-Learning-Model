
# Student Performance Prediction using Machine Learning

## Project Overview

This project focuses on predicting student final grades using Machine Learning techniques.  
The Student Performance Dataset was analyzed and preprocessed before applying machine learning algorithms.

Two regression models were used:
- Linear Regression
- Random Forest Regressor

The project helps understand how factors such as study time, attendance, failures, and previous grades influence student academic performance.

---

## Objective

- Load and preprocess student data
- Encode categorical variables
- Apply feature scaling
- Train machine learning models
- Predict final student grades
- Compare model performance
- Visualize prediction results

---

## Dataset Information

### Dataset Name
Student Performance Dataset

### Features Used

| Column Name | Description |
|---|---|
| school | School name |
| sex | Gender |
| age | Student age |
| address | Address type |
| famsize | Family size |
| Pstatus | Parent status |
| Medu | Mother's education |
| Fedu | Father's education |
| studytime | Weekly study time |
| failures | Past class failures |
| internet | Internet access |
| absences | Number of absences |
| G1 | First period grade |
| G2 | Second period grade |
| G3 | Final grade (Target Variable) |

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Machine Learning Models Used

### 1. Linear Regression
Predicts student performance using a linear relationship between features and target values.

### 2. Random Forest Regressor
Uses multiple decision trees to improve prediction accuracy.

---

## Project Workflow

1. Load dataset
2. Clean column names
3. Handle missing values
4. Encode categorical data
5. Select features and target
6. Apply feature scaling
7. Split data into training and testing sets
8. Train machine learning models
9. Evaluate models using RMSE
10. Visualize prediction results
11. Save output files

---

## Model Evaluation

The models were evaluated using:

- RMSE (Root Mean Squared Error)

Lower RMSE values indicate better prediction performance.

---

## Visualization

A scatter plot was created to compare:
- Actual G3 values
- Predicted G3 values

The graph helps analyze prediction accuracy visually.

---

## Output Files

| File Name | Description |
|---|---|
| ml_output.csv | Predicted output values |
| prediction_plot.png | Actual vs Predicted graph |

---

## Insights

- Previous grades strongly influence final grades.
- Study time and attendance affect academic performance.
- Random Forest generally provides better prediction accuracy.
- Machine learning helps identify students needing academic support.

---

## Conclusion

This project successfully implemented machine learning models to predict student performance.  
The analysis demonstrated how preprocessing, feature scaling, and regression models can be used for educational analytics and prediction tasks.

---

## Future Enhancements

- Hyperparameter tuning
- Advanced regression models
- Interactive dashboards
- Real-time prediction systems
- Web application deployment

---

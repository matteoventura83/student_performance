# Analysing student profiles and behaviours to support personalised educational strategies

## Overview & Objectives

This project is the result of my assignment for the Data Mining & text Analysis module of my MSc in Computer Science with Data Analytics. It applies data mining techniques to investigate two key areas of educational data mining: 
<ul><li>Identifying the factors that influence students' final grades</li>
<li>Detecting students who are at risk of poor academic performance</li>  </ul>

The overall aim is to use data-driven methods to better understand student characteristics and behaviours, identify underlying patterns, and support the development of targeted educational strategies.

## Dataset

For this analysis, the dataset provided was <i>student_performance.csv</i>, which contains records from two subjects, Mathematics and Portuguese, with 34 features collected from students attending two public schools. The dataset includes demographic, socio-economic, and academic variables, as well as categorical variables such as guardian type, reason for school choice, and parental occupation. 

Students are assessed three times during the school year. The features <i>G1</i> and <i>G2</i> represent the first and second assessment periods, while <i>G3</i> represents the final grade.

## Language & Libraries

<b>Language</b>: Python 3<br><br>
<b>Libraries</b>: 
<ul><li><b>Pandas</b> for data cleaning, manipulation, and analysis</li>
<li><b>NumPy</b> for handling numerical operations and array structures</li>
<li><b>Matplotlib</b> and <b>Seaborn</b> for data visualization and creation of correlation heatmaps</li>
<li><b>Scikit-learn</b> for performing regression analysis, feature scaling, and model evaluation</li></ul>

## Results

### Problem 1: which socio-economic, demographic, and academic factors most strongly influence the final grade (G3)?

My initial assumption was that family situation could affect student performance, as grades cannot be explained solely from an academic perspective but need to consider different aspects of students’ lives.

The exploratory data analysis showed that the dataset is clean, as no missing values, duplicate entries, invalid data types or outliers indicating potential data entry errors were detected. Only standardization of the column names was performed.

Supervised learning models were applied because the dataset contains labelled data, where several independent variables representing students’ characteristics were used to train the algorithm to predict the target variable, <i>g3</i>, which represents the final grade. However, as these models require numerical input, categorical variables were converted into numerical format during the preprocessing stage.

Multiple Linear Regression was used to train the model using students’ characteristics to predict the g3 variable, explaining how the independent variables affect the target variable by providing coefficients for each predictor. 

Before training the Multiple Linear Regression model, as the numerical independent variables covered distinctly different ranges, it was necessary to standardize them, transforming them to a similar scale so that the model could learn appropriate weights for each feature. In this way, variables with larger numerical ranges do not disproportionately influence the model, ensuring a more balanced contribution of each predictor and improving the accuracy of the coefficient estimates.

Secondly, the Variance Inflation Factor (VIF) metric was used to detect variables that are highly correlated with each other, which may cause multicollinearity. To address this issue, all variables with a VIF score higher than 10 were excluded, and one category from each one-hot encoded categorical variable was also removed.

However, the results were not satisfactory. The R² score, used to determine the explanatory power of the model, was 0.10, indicating that the model captured only a small portion of the variability in the data and therefore had limited effectiveness. The Mean Absolute Error (MAE) was 2.63. As 70.88% of <i>g3</i> values are concentrated between 8 and 12, this number represents a substantial deviation relative to the effective range of the data. The Mean Squared Error (MSE), which penalizes large errors more heavily, was 13.89, suggesting that the model’s predictions deviate further from the true values, indicating poorer performance. 

As a test, a model was run including all the previously excluded independent variables, and it was observed that the R² score was very high (0.80), while the MAE (1.02) and MSE (3.10) were very low, indicating a model capable of accurately predicting the target variable. However, as shown in Figure 1, analysis of the coefficients revealed that g2 clearly dominates all other variables. 

<img width="580" height="386" alt="the top variables influencing g3 when using Multiple Linear Regression" src="https://github.com/user-attachments/assets/f785fc10-e9b5-499a-bca8-2ebf8e60c13c" />

Figure 1: the top variables influencing g3 when using Multiple Linear Regression

Decision Tree Regressor was applied to predict the target variable <i>g3</i>. This model can capture more complex, non-linear relationships without requiring normalization of the independent variables, while maintaining interpretability through a clear set of decision rules and feature importance measures that indicate the influence of each predictor. Furthermore, it can handle multicollinearity effectively due to its inherent nature of recursive splitting and feature selection.

The results are very satisfactory. The R² score was 0.84, indicating the model was capable to predict the target variable. This is also confirmed by low MAE (0.93) and MSE (1.56), confirming a low margin of error compared to the target variable. 

Even though the R² score was already high, cross-validation was applied to provide a more robust estimate of the model’s performance. This procedure evaluates the average performance across multiple training and validation splits, further reducing the risk of overfitting and assessing whether the model performs consistently across different subsets of the data. However, the R² score remained at a similar level (0.82), confirming the stability of the model. As shown in Figure 2, when examining the feature importances, the academic variable g2 showed an outstanding influence (0.92), clearly dominating all other predictors.

<img width="991" height="584" alt="image" src="https://github.com/user-attachments/assets/5326048b-d195-463b-91f3-388c0f8f68f3" />
Figure 2: the top variables influencing g3 when using Decision Tree Regressor

In conclusion, although both models showed high predictive accuracy, an important limitation remains for explanatory analysis, as the strongest predictor, <i>g2</i>, dominated the results, masking the influence of socio-economic and demographic variables, which may still affect performance but are not effectively captured by the models.






### Problem 2: detecting students at risk of underperformance and disengagement.


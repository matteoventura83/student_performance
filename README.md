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

Multiple Linear Regression was used to train the model using students’ characteristics to predict the <i>g3</i> variable, explaining how the independent variables affect the target variable by providing coefficients for each predictor. 

Before training the Multiple Linear Regression model, as the numerical independent variables covered distinctly different ranges, it was necessary to standardize them, transforming them to a similar scale so that the model could learn appropriate weights for each feature. In this way, variables with larger numerical ranges do not disproportionately influence the model, ensuring a more balanced contribution of each predictor and improving the accuracy of the coefficient estimates.

Secondly, the Variance Inflation Factor (VIF) metric was used to detect variables that are highly correlated with each other, which may cause multicollinearity. To address this issue, all variables with a VIF score higher than 10 were excluded, and one category from each one-hot encoded categorical variable was also removed.

However, the results were not satisfactory. The R² score, used to determine the explanatory power of the model, was 0.10, indicating that the model captured only a small portion of the variability in the data and therefore had limited effectiveness. The Mean Absolute Error (MAE) was 2.63. As 70.88% of <i>g3</i> values are concentrated between 8 and 12, this number represents a substantial deviation relative to the effective range of the data. The Mean Squared Error (MSE), which penalizes large errors more heavily, was 13.89, suggesting that the model’s predictions deviate further from the true values, indicating poorer performance. 

As a test, a model was run including all the previously excluded independent variables, and it was observed that the R² score was very high (0.80), while the MAE (1.02) and MSE (3.10) were very low, indicating a model capable of accurately predicting the target variable. However, as shown in Figure 1, analysis of the coefficients revealed that <i>g2</i>clearly dominates all other variables. 

<img width="580" height="386" alt="the top variables influencing g3 when using Multiple Linear Regression" src="https://github.com/user-attachments/assets/f785fc10-e9b5-499a-bca8-2ebf8e60c13c" />

Figure 1: the top variables influencing <i>g3</i> when using Multiple Linear Regression

Decision Tree Regressor was applied to predict the target variable <i>g3</i>. This model can capture more complex, non-linear relationships without requiring normalization of the independent variables, while maintaining interpretability through a clear set of decision rules and feature importance measures that indicate the influence of each predictor. Furthermore, it can handle multicollinearity effectively due to its inherent nature of recursive splitting and feature selection.

The results are very satisfactory. The R² score was 0.84, indicating the model was capable to predict the target variable. This is also confirmed by low MAE (0.93) and MSE (1.56), confirming a low margin of error compared to the target variable. 

Even though the R² score was already high, cross-validation was applied to provide a more robust estimate of the model’s performance. This procedure evaluates the average performance across multiple training and validation splits, further reducing the risk of overfitting and assessing whether the model performs consistently across different subsets of the data. However, the R² score remained at a similar level (0.82), confirming the stability of the model. As shown in Figure 2, when examining the feature importances, the academic variable <i>g2</i> showed an outstanding influence (0.92), clearly dominating all other predictors.

<img width="580" height="386" alt="the top variables influencing g3 when using Decision Tree Regressor" src="https://github.com/user-attachments/assets/5326048b-d195-463b-91f3-388c0f8f68f3" />

Figure 2: the top variables influencing <i>g3</i> when using Decision Tree Regressor

In conclusion, although both models showed high predictive accuracy, an important limitation remains for explanatory analysis, as the strongest predictor, <i>g2</i>, dominated the results, masking the influence of socio-economic and demographic variables, which may still affect performance but are not effectively captured by the models.


### Problem 2: detecting students at risk of underperformance and disengagement.

The second problem addressed was the identification of at-risk students. My initial assumption was that underperformance could be identified by considering the variables <i>g3</i> and <i>failures</i>, while <i>study_time</i> and <i>absences</i> could be used to assess disengagement, since the time devoted to studying and school attendance are strong indicators of engagement. 

An unsupervised learning model was considered appropriate because, as the dataset does not contain variables explicitly indicating which students are underperforming or disengaged, it helps identify patterns without the guidance of labelled outputs. Clustering is an unsupervised technique which is particularly suitable because, by extracting patterns from the dataset, it groups unlabelled data with similar characteristics into distinct clusters. Among the several clustering techniques, K-Means was chosen because it aims to ensure that data points within a cluster are as close as possible to each other, ensuring high cohesion, while simultaneously maximizing the separation between clusters.

As previously noted, the exploratory data analysis showed that the dataset is clean. Although three students exhibited very high numbers of absences, they were not excluded because these values were realistic and such cases are common in schools.

As the four variables considered are numerical and the K-Means algorithm requires standardization to produce accurate results, StandardScaler was applied. Since K-Means is a distance-based algorithm and can be sensitive to the scale of the features, standardization ensures that all variables are treated equally.

To detect the group of students at risk of underperformance, the features <i>g3</i> and failures were selected. The elbow method indicated that the appropriate number of clusters was five. As shown in Figure 3, cluster 2 is characterised by low grades and the highest number of failures. This was also confirmed by a groupby analysis, which showed that cluster 2 had the second lowest mean <i>g3</i> grade (6.80) and the highest mean number of failures (2.50).

<img width="416" height="325" alt="cluster 2 shows the worst performance in terms of failures and grades" src="https://github.com/user-attachments/assets/659178b3-7a52-4ede-bf7f-5f4cf6ac5622" />

Figure 3: cluster 2 shows the worst performance in terms of failures and grades

To detect the group of students at risk of disengagement, the features <i>study_time</i> and <i>absences</i> were selected. The elbow method indicated that the appropriate number of clusters was four. Figure 4 indicates that cluster 2 is characterised by very low study time and the highest number of absences. The groupby analysis confirmed this result, as cluster 2 showed the second lowest mean study time (1.70) and the highest mean number of absences (17.94).

<img width="580" height="269" alt="image" src="https://github.com/user-attachments/assets/24619a2d-e33c-447f-8289-a2ee17937e6e" />

Figure 4: cluster 2 indicates disengagement, reflected in both failures and grades

To assess the models, the Silhouette Score was used because it measures how similar an object is to its own cluster compared with other clusters. This metric is easy to interpret, as it ranges from −1 to 1, with values closer to 1 indicating better-defined clusters. Furthermore, it provides insight into cluster cohesion and separation.

The Davies–Bouldin index was also used, as it measures the size of clusters against the average distance between clusters. This metric is easy to interpret as it uses point-wise distances, and a lower score indicates better clustering. 

Both metrics confirmed positively the reliability and stability of the clustering, indicating that the clusters effectively identify groups of students at risk: for the underperformance clustering, the Silhouette Score was 0.578 and the Davies–Bouldin Index was 0.596, while for the disengagement clustering the Silhouette Score was 0.555 and the Davies–Bouldin Index was 0.682. 

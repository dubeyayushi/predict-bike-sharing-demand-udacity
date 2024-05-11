# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Ayushi Dubey

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
While submitting predictions, I encountered several challenges including syntax errors and the need to acquaint myself with SageMaker's environment. Initially, linking my Kaggle account posed difficulties, and the submission's initial score exceeded 1, indicating poor performance. To improve this, I refined feature engineering and conducted hyperparameter tuning. Additionally, I had to ensure predictions didn't contain negative values, as Kaggle rejects submissions with negatives. These challenges demanded iterative adjustments to my approach and increased familiarity with Kaggle competitions and SageMaker's functionalities.

### What was the top ranked model that performed?
The top-ranked model was the (add features) model named WeightedEnsemble_L3, with the best Kaggle score of 0.44957 (on test dataset). This model was developed by training on data obtained using exploratory data analysis (EDA) and feature engineering without the use of a hyperparameter optimization routine.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
The exploratory analysis revealed that the datetime feature provided valuable temporal information, allowing for the extraction of additional features such as year, month, day (dayofweek), and hour. These extracted features were treated as distinct independent variables, enhancing the model's ability to capture temporal patterns. Furthermore, to accurately represent categorical variables like season and weather, which were initially interpreted as integers by AutoGluon, the dtype was set to "category". This adjustment ensured that AutoGluon treated these variables appropriately during model training, capturing their categorical nature effectively.

### How much better did your model preform after adding additional features and why do you think that is?
The model showed a significant improvement in performance after adding additional features. Initially, with no additional features and default settings, the model achieved a score of 1.84007. However, after incorporating additional features, the model's performance substantially improved, achieving a score of 0.44957. This represents a reduction in error by approximately 1.3905. The improvement can be attributed to the increased richness of the feature set, allowing the model to capture more nuanced patterns and relationships within the data. 

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
After performing hyperparameter tuning (HPT), the model's performance experienced a slight degradation compared to the best performing model with the add features. However, despite this slight decrease in performance, the model still outperformed the initial model trained with default settings. This suggests that while some hyperparameter configurations may not result in significant improvements, the overall tuning process contributes positively to model performance by ensuring that the model is optimized for the specific characteristics of the dataset.

### If you were given more time with this dataset, where do you think you would spend more time?
Given more time to work with this dataset, I would like to investigate additional potential outcomes when AutoGluon is run for an extended period with a high quality preset and enhanced hyperparameter tuning.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|prescribed_values|prescribed_values|"presets: 'high quality' (auto_stack=True)"|1.84007|
|add_features|prescribed_values|prescribed_values|"presets: 'high quality' (auto_stack=True)"|0.44957|
|hpo|"{'num_boost_round': 100, 'num_leaves': 36, 'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}}" (for the GBM hyperparameters)|"{'num_epochs': 5, 'learning_rate': 0.0005, 'activation': 'relu', 'dropout_prob': 0.1}" (for the NN_TORCH hyperparameters)|"presets: 'optimize_for_deployment"|0.53849|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.

![model_train_score.png](nd009t-c1-intro-to-ml-project-starter/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![model_test_score.png](nd009t-c1-intro-to-ml-project-starter/model_test_score.png)

## Summary
In summary, conducted three different runs of model training and evaluation were conducted using AutoGluon's Tabular Predictions.

1. In the "initial" run, I trained the model with default hyperparameters and obtained a score of 1.84007 on Kaggle.
2. In the "add_features" run, I enhanced the model by adding additional features extracted from the datetime column and achieved a significantly improved score of 0.44957.
3. Lastly, in the "hpo" run, I performed hyperparameter optimization (HPO) to further refine the model's performance. Despite observing a slight degradation in score compared to the "add_features" run, the model still outperformed the initial model with a score of 0.53849.
Overall, through feature engineering, hyperparameter tuning, and iterative refinement, I successfully improved the model's predictive accuracy, demonstrating the effectiveness of these techniques in enhancing model performance for regression tasks.

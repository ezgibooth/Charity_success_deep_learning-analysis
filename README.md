# Deep Learning Anlaysis of Charity Success

## Background

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. 
Using machine learning and neural networks, we have used the features in the provided dataset to create a binary classifier that can predict whether applicants 
will be successful if funded by Alphabet Soup.




## Report on the Neural Network Model

### Overview of the analysis:

The goal of this analysis is to create a binary classifier using the provided dataset that can predict whether applicants will be successful if funded by Alphabet Soup.
<br>

Within this dataset are a number of columns that capture metadata about each organization, such as:

* EIN and NAME—Identification columns
* APPLICATION_TYPE—Alphabet Soup application type
* AFFILIATION—Affiliated sector of industry
* CLASSIFICATION—Government organization classification
* USE_CASE—Use case for funding
* ORGANIZATION—Organization type
* STATUS—Active status
* INCOME_AMT—Income classification
* SPECIAL_CONSIDERATIONS—Special considerations for application
* ASK_AMT—Funding amount requested
* IS_SUCCESSFUL—Was the money used effectively


### Results:

#### Data Preprocessing

* The target variable for this model was whether it was successful or not.
* The features that were used as variables for this model were
* The variables that were removed from the input data were:
  * EIN (does not contribute any information for creating a model)
  * NAME (initially removed but placed back as there were numerous institutions that had more than five applications and proved to be an indicator for success)
  * SPECIAL CONSIDERATIONS (of the two possible options for this column, less than 0.1% made up the Yes values, compared to the 99.9% consisting of No values)
  * STATUS ((of the two possible options for this column, less than 0.1% made up the Yes values, compared to the 99.9% consisting of No values)
* Furthermore, cutoff values were determined and used for 'APPLICATION TYPES' and 'CLASSIFICATION' and 'AFFLIATION' columns, to create an 'Other' column to consolidate very small values that made up less than 1% of the overall data

#### Compiling, Training, and Evaluating the Model

For each model, unique values for each column of the dataset were determined. For columns that have more than 10 unique values, the number of data points for each unique value was determined. The number of data points for each unique value was used to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful. Pd.get_dummies() was used to encode categorical variables. The preprocessed data was split into a features array, X, and a target array, y. These arrays and the train_test_split function was used to split the data into training and testing datasets. The training and testing features datasets were scaled by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

* Initial Model: 
  * 'EIN' and 'NAME' columns were dropped
  * Application types value cutoff set at <200
  * Classiciation counts set at <1800
  * Two hidden layers, with 90 and 30 nodes respectively with 'relu' activation were used
  * Analysis showed the model had 0.729 accuracy
* Optimizated Model 1:
  * In addition to 'EIN' and 'NAME', 'Special considerations' column was dropped
  * A third hidden layer was added to the nn model
  * Epochs value was set to 50, as more did not seem to make a difference in the accuracy
  * Three hidden layers, with 80, 40 and 20 nodes respectively with 'relu' activation were used
  * Analysis showed the model had 0.729 accuracy
* Optimizated Model 2:
  * In addition to 'EIN' and 'NAME', 'Special considerations' column was dropped
  * Affliation sypes were modified to have only independent, company and others
  * A forth hidden layer was added to the nn model
  * Epochs value was set to 50, as more did not seem to make a difference in the accuracy
  * Four hidden layers, with 30, 30, 30 and 20 nodes respectively with 'tanh' activation were used
  * Analysis showed the model had 0.731 accuracy
* Optimizated Model 3:
  * In addition to 'EIN', 'NAME', and 'Special considerations', 'Status' column was also dropped
  * Affliation sypes were modified to have only independent, company and others
  * A forth hidden layer was added to the nn model
  * Epochs value was set to 20, as more did not seem to make a difference in the accuracy
  * Four hidden layers, with 30, 30, 30 and 20 nodes respectively with 'tanh' activation were used
  * Analysis showed the model had 0.734 accuracy
* Optimizated Model 4:
  * Using the previous modified dataset, this model used two hidden layers 
  * Two hidden layers, with 37 and 20 nodes with 'relu activation' were used
  * Analysis showed the model had 0.734 accuracy
* Final Optimizated Model:
  * All the above models with varying hidden layers, nodes and activation types resulted in accuracy less than 0.75
  * Consequently, in this final model, 'NAME' column was not dropped as it had valuable infomration on repeat borrowers.
  * In 'NAME' column, those with value counts less than 5 were bundled into 'other' category
  * KERAS model was used to determine the optimal model
  * The final model used has three hidden layers with 25, 30 and 10 nodes respectively with 'tanh' activation and epoch set to 100
  * Anlaysis showed the model had 0.793 accuracy

### Summary: 

Models that used the dataset without the NAME column, were not able achieve the desired 0.75 accuracy regardless of models, number of hidden layers or neurons used. Noticing and including the 'NAME' column made a significant difference in the training of the data, and the final optimized model was able to achieve 0.79 accuracy. This project was important in understanding how crucial it is to include all relevant data to train a model and accurately predit results, and without the right datasets, the neural networks model by itself is not sufficient. Additionally, if time allowed, it would have been good to consider other top performing models that the Keras model showed to see whether accurcary could be increased.

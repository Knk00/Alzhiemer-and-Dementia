Dementia Predictive Analysis :
This is a project based on the prediction of whether a person, given the dataset, is demented or not.
The dataset has been taken from OASIS-brains.org (Open Access Series of Imaging Studies). 
OASIS has a collection of datasets on MRI scans taken volunteers.
I have used OASIS-longitudnal dataset to predict the level of dementia in the given subjects.
Pre-Processing techniques like handling of missing values, label encoding, one hot encoding and other methods have been used.
Model Selection was done using k-fold cross-validation.
The model that I chose was the Rainforest Classifier, from the sklearn.ensemble package, to fit the model on the dataset.
Algorithm Tuning was done using GridSearchCV.
Prediction was calculated on the scoring method of accuracy, shown by Confusion Matrix, Classification report and OLS regression results.
Finally, visualization of the prediction and correlation was done using Heatmaps and Boxenplot from the Seaborn library.

Thank you, for going through the Repo. Hope it helped you in someway or the other.

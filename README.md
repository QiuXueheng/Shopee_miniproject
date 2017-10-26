# Shopee_miniproject
1. Data Preprocessing:

    i). Please make sure the original csv files are located in a folder named 'Shopee' under the main directory.

    ii). Run 'GenerateTrainingData.py'. After the feature organization procedures are completed, the file 'traindata.csv' would be created     in the main directory.

2. Learning Model Training:

    i). Run 'sklearnModelTraining.csv'. Three kinds of classifiers would be trained with fine-tuned hyperparameters: logistic Regression,     Desicion Tree and Random Forests. The trained model would be saved in the folder "Models".

    ii). Run 'kerasModelTraining.csv'. Single hidden layer feedforward neural networks and LSTMs would be trained and saved.

3. Prediction:

    i). Run 'Prediction' to generate the predicted labels for the test samples in the file 'predict.csv'. The results would be saved in a     file named 'predict_result.csv' in the main directory

    ii). In the code, set the variable 'pathselect' as 0 to use pretrained models in the folder 'Pretrained_models', or set as 1 to use       the generated models in the folder 'models'

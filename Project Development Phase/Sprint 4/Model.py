# Importing essential libraries
import numpy as np
import pandas as pd
import pickle
import warnings

# Loading the dataset
df = pd.read_csv('./Data/Admission_Predict.csv')


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


df = clean_dataset(df)




# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN



df_copy = df.copy(deep=True)
df_copy[['GREScore','TOEFLScore','UniversityRating','SOP','LOR ','CGPA','Research','ChanceofAdmit ']] = df_copy[['GREScore','TOEFLScore','UniversityRating','SOP','LOR ','CGPA','Research','ChanceofAdmit ']].replace(0,np.NaN)

df_copy['GREScore'].fillna(df_copy['GREScore'].median(),inplace=True)
df_copy['TOEFLScore'].fillna(df_copy['TOEFLScore'].median(),inplace=True)
df_copy['UniversityRating'].fillna(df_copy['UniversityRating'].median(),inplace=True)
df_copy['SOP'].fillna(df_copy['SOP'].median(),inplace=True)
df_copy['LOR '].fillna(df_copy['LOR '].median(),inplace=True)

df_copy['CGPA'].fillna(df_copy['CGPA'].median(),inplace=True)

df_copy['Research'].fillna(df_copy['Research'].median(),inplace=True)
df_copy['ChanceofAdmit '].fillna(df_copy['ChanceofAdmit '].median(),inplace=True)

# Replacing NaN value by mean, median depending upon distribution





X=df_copy.drop('ChanceofAdmit ',axis=1)
Y=df_copy['ChanceofAdmit ']




from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape



from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor (n_estimators =1000, max_depth = 10, random_state = 34)


regressor.fit (X_train, np.ravel(y_train, order = 'C'))




# Creating a pickle file for the classifier
filename = 'Model/prediction-rfc-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))




filename = 'Model/prediction-rfc-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))

filename = 'Model/prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))


data = np.array([[337,118,4,4.5,4.5,9.65,1]])
my_prediction = classifier.predict(data)
warnings.filterwarnings("ignore", category=DeprecationWarning)
print(my_prediction[0])




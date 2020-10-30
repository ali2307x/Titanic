import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
# from matplotlib import pyplot as plt

data = pd.read_csv('Data/train.csv', delimiter=',', quotechar='"', skipinitialspace=True)
full_data = pd.read_csv('Data/full.csv', delimiter=',', quotechar='"', skipinitialspace=True)


# Return Sex and Embarked columns as binarized numpy array
class CatTransformer(BaseEstimator, TransformerMixin):

    sex_encoder = LabelEncoder()
    embark_encoder = LabelBinarizer()

    def fit(self, X, y=None):
        sex_1hot = self.sex_encoder.fit(X['Sex'])
        embarks_1hot = self.embark_encoder.fit(X['Embarked'])
        return self

    def transform(self, X, y=None):
        sex_1hot = self.sex_encoder.transform(X['Sex'])
        embarks_1hot = self.embark_encoder.transform(X['Embarked'])
        return np.c_[sex_1hot, embarks_1hot]


# Fill out the missing values for the Age column
class AgeTransformer(BaseEstimator, TransformerMixin):

        age_regressor = None
        age_scaler = StandardScaler()
        mode = ''
        mean = 0

        def __init__(self, mode='train'):
            self.mode = mode

        def fit(self, X, y=None):
            X = X[~X['Age'].isnull()]
            y = X['Age']
            X = X.drop(['Age', 'Cabin'], axis=1)
            self.age_regressor = LinearRegression()
            self.age_regressor.fit(X, y)
            return self

        def transform(self, X, y=None):
            if self.mode == 'train':
                self.mean = X['Age'].mean()
            features = X.drop(['Age', 'Cabin'], axis=1)
            for i in range(X.shape[0]):
                if np.isnan(X.at[i, 'Age']):
                    temp = features.iloc[i].values
                    if self.mode == 'test':
                        x = np.random.random_integers(0, 1, 1)
                        temp = np.insert(temp, 0, x)
                    new_age = self.age_regressor.predict([temp])[0]
                    X.at[i, 'Age'] = new_age if new_age > 0 else self.mean
            if self.mode == 'train':
                X['Age'] = self.age_scaler.fit_transform(X['Age'].to_numpy().reshape(-1, 1))
            elif self.mode == 'test':
                X['Age'] = self.age_scaler.transform(X['Age'].to_numpy().reshape(-1, 1))
            return X


# Convert the Cabin column into columns Cabin Letter and Number
class CabinTransformer(BaseEstimator, TransformerMixin):

        def extract_letter_and_number(self, cabin):
            if len(cabin) == 1:
                return cabin, 0
            else:
                cabin_split = cabin.split(' ')
                picked_cabin = cabin_split[0] if len(cabin_split[0]) > 1 else cabin_split[-1]
                return picked_cabin[0], np.int(picked_cabin[1:])

        def letters_and_numbers_cols(self, cabin_col):
            cabin_letters, cabin_numbers = [], []
            for cabin in cabin_col:
                if cabin is not np.NaN:
                    cabin_letter, cabin_number = self.extract_letter_and_number(cabin)
                else:
                    cabin_letter, cabin_number = np.NaN, np.NaN
                cabin_letters.append(cabin_letter)
                cabin_numbers.append(cabin_number)

            return cabin_letters, cabin_numbers

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            cabin_letters, cabin_numbers = self.letters_and_numbers_cols(X['Cabin'])
            X['CabinLetters'] = cabin_letters
            X['CabinNumbers'] = cabin_numbers
            return X.drop(['Cabin'], axis=1)


# Fill out the missing values for the Cabin column
class CabinPredictor(BaseEstimator, TransformerMixin):

        cabin_predictor = None
        cabin_letter_encoder = LabelBinarizer()
        mode = 'train'

        def fit(self, X, y=None):
            X = X[~X['CabinLetters'].isnull()]
            y = np.c_[X['CabinLetters'], X['CabinNumbers'].astype(int).astype(str)]
            X = X.drop(['CabinLetters', 'CabinNumbers'], axis=1)
            self.cabin_predictor = KNeighborsClassifier()
            self.cabin_predictor.fit(X, y)
            return self

        def transform(self, X, y=None):

            for i, row in X.iterrows():
                if X.at[i, 'CabinLetters'] is np.NaN:
                    temp = X.loc[i].values[:-2]  # do not take cabin letter and number column values

                    if 'Survived' not in X.columns:
                        x = np.random.random_integers(0, 1, 1)
                        temp = np.insert(temp, 0, x)

                    predict = self.cabin_predictor.predict([temp])[0]

                    X.at[i, 'CabinLetters'] = predict[0]
                    X.at[i, 'CabinNumbers'] = predict[1]

            if self.mode == 'train':
                cabin_letters_col_1hot = self.cabin_letter_encoder.fit_transform(X['CabinLetters'])
            else:
                cabin_letters_col_1hot = self.cabin_letter_encoder.transform(X['CabinLetters'])

            X.drop(['CabinLetters'], axis=1, inplace=True)
            return np.c_[X, cabin_letters_col_1hot]


# Change the order of the P class from [1, 2, 3] to [3, 2, 1]
class ClassCorrector(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for i in range(X.shape[0]):
            if X.at[i, 'Pclass'] == 1:
                X.at[i, 'Pclass'] = 3
            elif X.at[i, 'Pclass'] == 3:
                X.at[i, 'Pclass'] = 1
        return X


# Fill missing Embarked values
# Drop Passenger ID and Name columns
# Insert new column family size
# Scale the Fare column
# Correct the P Class
class FirstStep(BaseEstimator, TransformerMixin):

    fair_scaler = StandardScaler()
    class_correct = ClassCorrector()

    def fit(self, X, y=None):
        self.fair_scaler.fit(X['Fare'].to_numpy().reshape(-1, 1))
        return self

    def transform(self, X, y=None):
        X['Embarked'].fillna('S', inplace=True)
        X.drop(['PassengerId', 'Name'], axis=1, inplace=True)
        familysize = X['SibSp'] + X['Parch']
        X.insert(X.columns.get_loc('Parch') + 1, 'FamilySize', familysize)

        X['Fare'] = self.fair_scaler.transform(X['Fare'].to_numpy().reshape(-1, 1))
        X = self.class_correct.transform(X)
        return X


# Return a binarized numpy array of the tickets
class TicketTransformer(BaseEstimator, TransformerMixin):

    ticket_binarizer = LabelBinarizer()

    def fit(self, X, y=None):
        ticket_col = full_data['Ticket'].apply(lambda x: 'NUMBER' if x.isdigit() else x)
        ticket_col = ticket_col.apply(
            lambda x: x.replace('/', '').replace('.', '').replace('STONO 2', 'STONO2').split(' ')[0])
        self.ticket_binarizer.fit(ticket_col)
        return self

    def transform(self, X, y=None):
        ticket_col = X['Ticket'].apply(lambda x: 'NUMBER' if x.isdigit() else x)
        ticket_col = ticket_col.apply(
            lambda x: x.replace('/', '').replace('.', '').replace('STONO 2', 'STONO2').split(' ')[0])
        ticket_col = self.ticket_binarizer.transform(ticket_col)
        return ticket_col


first_step = FirstStep()
data = first_step.fit_transform(data)

cat_trans = CatTransformer()
ticket_trans = TicketTransformer()

cat_cols = cat_trans.fit_transform(data)  # Sex and Embarked binarized columns
data['Sex'] = cat_cols[:, 0]
data.drop(['Embarked'], axis=1, inplace=True)

ticket_cols = ticket_trans.fit_transform(data)  # Binarized ticket column
data.drop(['Ticket'], axis=1, inplace=True)


age_reg = AgeTransformer(mode='train')
data: pd.DataFrame = age_reg.fit_transform(data)

cabin_tran = CabinTransformer()
data = cabin_tran.fit_transform(data)

cabin_pred = CabinPredictor()
data: np.ndarray = cabin_pred.fit_transform(data)  # Shuffled Rows


# Do not include the sex column because it is already included
data = np.c_[data, cat_cols[:, 1:], ticket_cols]


X = data[:, 1:]  # Take everything but survived column
y = data[:, 0]

# param_grid = [
#                 {'n_estimators': [400], 'max_depth': [None, 7, 10, 13, 15], 'criterion': ['gini', 'entropy'],
#                  'max_features': ['auto', None, 3, 5], 'n_jobs': [-1]}
# ]
#
#
# rnd_clf = RandomForestClassifier()
# rnd_clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
#             max_depth=13, max_features=None, max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=700, n_jobs=-1,
#             oob_score=True, random_state=None, verbose=0,
#             warm_start=False)
#
#
# grid_search = GridSearchCV(rnd_clf, param_grid, cv=5, scoring="accuracy")
# grid_search.fit(X, y)
# print(grid_search.best_estimator_)
# rnd_clf = grid_search.best_estimator_
#
# scores = cross_val_score(rnd_clf, X, y, scoring="accuracy", cv=5)
# print(np.mean(scores))
#
# rnd_clf.fit(X, y)
# print(rnd_clf.oob_score_)

tree = DecisionTreeClassifier(criterion='entropy', max_depth=13)

scores = cross_val_score(tree, X, y, scoring="accuracy", cv=5)
print(np.mean(scores))
tree.fit(X, y)

joblib.dump(tree, "Decision Tree.pkl")

loaded_model = joblib.load("Decision Tree.pkl")

test_data = pd.read_csv('Data/test.csv', delimiter=',', quotechar='"', skipinitialspace=True)
ids = test_data['PassengerId']

test_data = first_step.transform(test_data)
cat_cols = cat_trans.transform(test_data)
test_data['Sex'] = cat_cols[:, 0]
test_data.drop(['Embarked'], axis=1, inplace=True)

ticket_cols = ticket_trans.transform(test_data)
test_data.drop(['Ticket'], axis=1, inplace=True)


age_reg.mode = 'test'
test_data: pd.DataFrame = age_reg.transform(test_data)
test_data = cabin_tran.transform(test_data)

cabin_pred.mode = 'test'
test_data: np.ndarray = cabin_pred.transform(test_data)

# Do not include the sex column because it is already included
test_data = np.c_[test_data, cat_cols[:, 1:], ticket_cols]

X = test_data

preds = loaded_model.predict(X).astype(int)

ids = ids.to_numpy().astype(int)
result = np.c_[ids, preds]
# print(result)

np.savetxt("Result.csv", result, delimiter=",")

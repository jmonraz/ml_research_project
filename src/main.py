"""
Machine Learning
Titanic Research Assignment
By: Jorge Monraz
"""
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import learning_curve
from sklearn.linear_model import Ridge

if __name__ == "__main__":

    data_path = './data'
    out_path = './output'
    file_path = 'titanic_passengers.csv'
    path = os.path.join(data_path, file_path)

    df = pd.read_csv(path)
    df = pd.read_csv(path)

    # perform EDA
    out_features = ['Pclass', 'Sex',
                    # 'Age',
                    #           'SibSp',
                    #           'Parch',
                    'Fare', 'Embarked']
    in_features = ['Survived']
    df = df[out_features+in_features].copy()

    # fill na using pandas
    df['Embarked'] = df['Embarked'].fillna(value='S')

    # one hot
    enc = LabelBinarizer()
    one_hot = enc.fit_transform(df['Sex'])

    # male(1), female(0)
    df['Sex'] = one_hot

    # one hot embarked feature
    one_hot = enc.fit_transform(df['Embarked'])
    df = pd.concat(
        [df, pd.DataFrame(one_hot, columns=['is_C', 'is_Q', 'is_S'])], axis=1)

    # drop Embarked column
    df = df.drop('Embarked', axis=1)

    # perform PCA
    features = df.loc[:, ['is_C', 'is_Q', 'is_S']]
    pca = PCA(n_components=1)
    features = pca.fit_transform(features)

    # add Embarked column and drop is_C, is_Q, and is_S column
    df['Embarked'] = features

    # drop columns
    df = df.drop(['is_C', 'is_Q', 'is_S'], axis=1)

    # draw correlation matrix
    fig, ax = plt.subplots()
    corr = df.corr()
    ax = sns.heatmap(corr, annot=True)
    path = os.path.join(out_path, 'titanic_passengers_correlation.png')
    plt.tight_layout()
    plt.savefig(path)

    # collinearity
    fig, ax = plt.subplots()
    corr = df.corr().abs()
    ax = sns.heatmap(corr, vmin=0.9, vmax=1, annot=True)
    path = os.path.join(out_path, 'titanic_passengers_collinearity.png')
    plt.savefig(path)

    # separating output and input
    X = df.loc[:, ['Pclass', 'Sex', 'Fare']]
    y = df.loc[:, 'Survived']

    # scale the data
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # separate data into training and testing, seed = 123321
    seed = 123321
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=seed, test_size=.1)

    # train
    # trying KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    # SelectKBest
    kselect = SelectKBest(k='all', score_func=f_classif)
    X_best = kselect.fit_transform(X, y)
    top_features_selected = kselect.get_support(indices=True)

    # draw a confusion matrix for the estimator
    fig, ax = plt.subplots()
    cmat = confusion_matrix(y, model.predict(X))
    path = os.path.join(out_path, 'titanic_passengers_confusion.png')
    ax = sns.heatmap(cmat, annot=True)
    plt.savefig(path)

    print(f'feature scores: {kselect.scores_}')
    print(f'train score: {score}')
    print(f'test score {test_score}')

    # learning curve
    sizes, t_scores, v_scores = learning_curve(
        KNeighborsClassifier(), X, y, train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0])

    t_scores_mean = np.mean(t_scores, axis=1)
    t_scores_std = np.std(t_scores, axis=1)

    v_scores_mean = np.mean(v_scores, axis=1)
    v_scores_std = np.std(v_scores, axis=1)

    fig, ax = plt.subplots()
    plt.plot(sizes, t_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(sizes, v_scores_mean, 'o-', color="g", label="Test score")
    plt.fill_between(sizes, t_scores_mean - t_scores_std,
                     t_scores_mean + t_scores_std, alpha=0.1, color="r")
    plt.fill_between(sizes, v_scores_mean - v_scores_std,
                     v_scores_mean + v_scores_std, alpha=0.1, color="g")
    ax.set(xlabel="Training Samples", ylabel='Score')
    ax.legend()
    ax.grid()
    path = os.path.join(out_path, 'titanic_learning_curve.png')
    plt.savefig(path)

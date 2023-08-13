from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def data():
    data = pd.read_csv("data/data_produced.csv", encoding='utf-8')
    vectorizer = CountVectorizer(decode_error="replace")
    vector = vectorizer.fit_transform(data['Email'].to_list())
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    transformer = TfidfTransformer()
    transformer.fit_transform(vector)
    joblib.dump(transformer, "models/transformer.pkl")

    vectorizer = joblib.load("models/vectorizer.pkl")
    transformer = joblib.load("models/transformer.pkl")
    vector = vectorizer.transform(data['Email'].to_list())
    tfidf = transformer.transform(vector)
    data.replace({'Label': {'ham': 0, 'spam': 1}}, inplace=True)
    Y = data['Label']
    X_train, X_test, Y_train, Y_test = train_test_split(tfidf.toarray(), Y, test_size=0.3, random_state=0, shuffle=True)
    df_tfidf = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names_out())
    non_zero_values = df_tfidf[df_tfidf != 0].stack()
    print(non_zero_values)
    return X_train, X_test, Y_train, Y_test

def bayes_train(X_train, Y_train):
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train, Y_train)
    joblib.dump(naive_bayes, "models/bayes.pkl")

def bayes_test(X_test, Y_test):
    naive_bayes = joblib.load("models/bayes.pkl")
    # predict_Y_train = naive_bayes.predict(X_train)
    # P = precision_score(Y_train, predict_Y_train, average='micro')
    # acc = accuracy_score(Y_train, predict_Y_train)
    # R = recall_score(Y_train, predict_Y_train, average='micro')
    # F1 = f1_score(Y_train, predict_Y_train, average='micro')
    # print('training set：P=', P, ' R=', R, ' F1=', F1, ' acc=', acc)
    predict_Y_test = naive_bayes.predict(X_test)
    P = precision_score(Y_test, predict_Y_test, average='binary')
    acc = accuracy_score(Y_test, predict_Y_test)
    R = recall_score(Y_test, predict_Y_test, average='binary')
    F1 = f1_score(Y_test, predict_Y_test, average='binary')
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(Y_test, predict_Y_test), display_labels=['Spam', 'Ham'])
    display.plot()
    plt.show()

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(Y_test, predict_Y_test)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()
    print('naive_bayes: acc =', acc, ' P =', P, 'R =', R, ' F1 =', F1)

def bayes_predict(x='Ya very nice. . .be ready on thursday'):
    vectorizer = joblib.load("models/vectorizer.pkl")
    transformer = joblib.load("models/transformer.pkl")
    vector = vectorizer.transform([x])
    tfidf = transformer.transform(vector)
    naive_bayes = joblib.load("models/bayes.pkl")
    y = naive_bayes.predict(tfidf.toarray())
    return y[0]

def random_forest_train(X_train, Y_train):
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_train, Y_train)
    joblib.dump(rf, "models/random_forest.pkl")

def random_forest_test(X_test, Y_test):
    rf = joblib.load("models/random_forest.pkl")
    predict_Y_test = rf.predict(X_test)
    P = precision_score(Y_test, predict_Y_test, average='binary')
    acc = accuracy_score(Y_test, predict_Y_test)
    R = recall_score(Y_test, predict_Y_test, average='binary')
    F1 = f1_score(Y_test, predict_Y_test, average='binary')

    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(Y_test, predict_Y_test), display_labels=['Spam', 'Ham'])
    display.plot()
    plt.show()

    fpr, tpr, _ = roc_curve(Y_test, predict_Y_test)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()
    print('Random_Forest: acc =', acc, ' P =', P, ' R =', R, ' F1 =', F1)

def random_forest_predict(x='Ya very nice. . .be ready on thursday'):
    vectorizer = joblib.load("models/vectorizer.pkl")
    transformer = joblib.load("models/transformer.pkl")
    vector = vectorizer.transform([x])
    tfidf = transformer.transform(vector)
    naive_bayes = joblib.load("models/random_forest.pkl")
    y = naive_bayes.predict(tfidf.toarray())
    return y[0]

def svm_train(X_train, Y_train):
    svm = SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
    svm.fit(X_train, Y_train)
    joblib.dump(svm, "models/svm.pkl")

def svm_test(X_test, Y_test):
    svm = joblib.load("models/svm.pkl")
    predict_Y_test = svm.predict(X_test)
    P = precision_score(Y_test, predict_Y_test, average='binary')
    acc = accuracy_score(Y_test, predict_Y_test)
    R = recall_score(Y_test, predict_Y_test, average='binary')
    F1 = f1_score(Y_test, predict_Y_test, average='binary')

    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(Y_test, predict_Y_test), display_labels=['Spam', 'Ham'])
    display.plot()
    plt.show()

    fpr, tpr, _ = roc_curve(Y_test, predict_Y_test)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()
    print('SVM: acc =', acc, ' P =', P, ' R =', R, ' F1 =', F1)

def svm_predict(x='Ya very nice. . .be ready on thursday'):
    vectorizer = joblib.load("models/vectorizer.pkl")
    transformer = joblib.load("models/transformer.pkl")
    vector = vectorizer.transform([x])
    tfidf = transformer.transform(vector)
    naive_bayes = joblib.load("models/svm.pkl")
    y = naive_bayes.predict(tfidf.toarray())
    return y[0]

if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = data()
    bayes_train(X_train, Y_train)
    bayes_test(X_test, Y_test)
    random_forest_train(X_train, Y_train)
    random_forest_test(X_test, Y_test)
    svm_train(X_train, Y_train)
    svm_test(X_test, Y_test)
    # print(bayes_predict())
    # print(random_forest_predict())
    # print(svm_predict())
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import pandas as pd
import numpy as np

#read in data
df = pd.read_csv("tumor_data_d.csv")
def duplicates():
#handle duplicates
    print("****************************************\n\n")
    print("print duplicates\n\n")
    duplicates=df.duplicated()

    print(duplicates)
    print("****************************************\n\n")

    #remove duplicates
    print(df[duplicates])
    df.drop_duplicates(inplace=True)
    print("****************************************\n\n")
    print("Verify duplicates are removed \n\n")
    df.info()
    print("****************************************\n\n")


def Imputation_and_Classification_models():

    #Imputation of data
    print("Impute missing data\n\n")
    X_num = df.drop(["Label"], axis=1).values
    y = df["Label"].values
    X_train_num, X_test_num, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=69,stratify=y)
    imp_num = SimpleImputer()
    X_train_num = imp_num.fit_transform(X_train_num)
    X_test_num = imp_num.transform(X_test_num)


    #Scale Data
    print("STANDARDIZING DATA...")
    scaler =StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train_num)
    X_test_scaled=scaler.transform(X_test_num)


    #Evaluating classification models
    models={"Logistic Regression":LogisticRegression(),"KNN":KNeighborsClassifier()}
    results=[]

    for model in models.values():
        kf = KFold(n_splits=5,random_state=69,shuffle=True)
        cv_results = cross_val_score(model,X_train_scaled,y_train,cv=kf)
        results.append(cv_results)


    #confusion Matrix
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled,y_train)
    y_knn_pred=knn.predict(X_test_scaled)
    print("KNN Confusion Matrix:")
    print(confusion_matrix(y_test,y_knn_pred))
    print("\nClassificaiton Report knn:")
    print(classification_report(y_test,y_knn_pred))
    print("\n")


    print("Logistic Regression Confusion Matrix:")
    logreg=LogisticRegression()
    logreg.fit(X_train_scaled,y_train)
    y_log_pred=logreg.predict(X_test_scaled)
    print(confusion_matrix(y_test,y_log_pred))
    print("\nClassification Report LogReg:")
    print(classification_report(y_test,y_log_pred))


    #Test set perfomance
    print("\nTest set Performances:")
    for name, model in models.items():
        model.fit(X_train_scaled,y_train)
        test_score = model.score(X_test_scaled,y_test)
        print("{} Test Set Accuracy : {}".format(name, test_score))

    plt.boxplot(results, labels=models.keys())
    plt.show()

def Imputation_and_Regression_models():
    #Imputation of data
    print("Impute missing data\n\n")
    X_num = df.drop(["Label","area_se"], axis=1).values
    y = df["area_se"].values
    X_train_num, X_test_num, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=69)
    imp_num = SimpleImputer()
    X_train_num = imp_num.fit_transform(X_train_num)
    X_test_num = imp_num.transform(X_test_num)


    #Scale Data
    print("STANDARDIZING DATA...")
    scaler =StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train_num)
    X_test_scaled=scaler.transform(X_test_num)

    #Evaluating classification models
    kf=KFold(n_splits=5,shuffle=True,random_state=69)
    reg=LinearRegression()
    cv_results=cross_val_score(reg,X_train_scaled,y_train,cv=kf)





print("Frankie China Quintero")
print("ACO423")
print("Module Phase 1 Supervised Learning")
print("****************************************\n\n")

    #information on data
df.info()
print(df)
print("****************************************\n\n")
print("find number of missing values \n")
print(df.isna().sum().sort_values())
print("****************************************\n\n")
duplicates()
Imputation_and_Classification_models()
print("****************************************\n\n")
print("Part B: Linear Regression:")
Imputation_and_Regression_models()



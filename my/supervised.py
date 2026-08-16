
import matplotlib.pyplot as plt 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#load data
df = pd.read_csv("wine.csv")

#null and duplicate remove
print(df.isnull().sum())
df = df.dropna()

print(df.duplicated().sum())
df = df.drop_duplicates()

X = df.drop("target", axis = 1)
y = df["target"]

#show data distribution
print("=====Data Distribution=====")
print(y.value_counts())
print("=====Show Percentage====")
print(y.value_counts(normalize = True) * 100)

#figure 1 , plot distribution
plt.figure()

y.value_counts().sort_index().plot(kind = "bar")

plt.title("wine distribution graph")
plt.xlabel("Target class")
plt.ylabel("number of samples")
plt.tight_layout()

#split
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 42
)

#scale
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

models = [
    ("Logistic Regression", LogisticRegression(max_iter = 1000)),
    ("SVM", SVC(kernel = "rbf")),
    ("Decision Tree", DecisionTreeClassifier(random_state = 42)),
    ("Random Forest", RandomForestClassifier(n_estimators = 100, random_state = 42)),
    ("KNN", KNeighborsClassifier(n_neighbors = 5)),
    ("Naive Bayes", GaussianNB())
]

model_names = []
accuracies = []

for name, model in models:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    acc = accuracy_score(y_test, pred)

    print("=====", name, "====")

    print("Accuracy : \n", acc)
    print("Confusion Matrix : \n", confusion_matrix(y_test, pred))
    print("Classification Report: \n", classification_report(y_test, pred))

    model_names.append(name)
    accuracies.append(acc)

plt.figure()
plt.bar(model_names, accuracies)

plt.title("model accuracy comparision")
plt.xlabel("ML models")
plt.ylabel("accuracy")
plt.xticks(rotation = 45)
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

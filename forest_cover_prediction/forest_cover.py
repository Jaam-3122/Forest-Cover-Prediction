import pandas as pd

df = pd.read_csv("train.csv")

print("\nMissing values:\n", df.isnull().sum())
print("\nTarget Class Distribution:\n", df['Cover_Type'].value_counts())

#dropping irrelevant column
df.drop(columns=['Id'], inplace=True)
df['Cover_Type'] = df['Cover_Type'] - 1

#split
X = df.drop('Cover_Type', axis=1)
y = df['Cover_Type']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Evaluation")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

#Fine tuning xgBoost
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1],
    'colsample_bytree': [0.7, 0.8, 1]
}

xgb = XGBClassifier(objective='multi:softmax', num_class=7, eval_metric='mlogloss')

random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_grid,
                                   n_iter=10, scoring='accuracy', cv=3, verbose=1, random_state=42)

random_search.fit(X_train, y_train)

print("Best Params:", random_search.best_params_)
best_xgb = random_search.best_estimator_

# Evaluate
y_pred = best_xgb.predict(X_test)
print("\nTuned XGBoost Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


#PCA 3D Scatter plot

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Apply PCA to reduce dimensions to 3
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='Set1', alpha=0.7)

ax.set_title("PCA Projection (3D) of Forest Terrain")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")

# Add legend
legend = ax.legend(*scatter.legend_elements(), title="Cover Type", loc="upper right")
ax.add_artist(legend)

plt.tight_layout()
plt.savefig("pca_3d_projection.png", dpi=300)
plt.show()

import joblib
joblib.dump(best_xgb, 'xgboost_forest_model.pkl')

model = joblib.load('xgboost_forest_model.pkl')
predictions = model.predict(X_test)

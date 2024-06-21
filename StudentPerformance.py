import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

data = pd.read_csv("student_performance_dataset.csv")
print(data.columns)
print(data.isnull().sum())
print(data.describe())
print(data.info())

# EDA
# distribution of GPA
plt.figure(figsize=(8, 6))
sns.histplot(data['GPA'], kde=True)
plt.title('Distribution of GPA')
plt.xlabel('GPA')
plt.ylabel('Frequency')
plt.show()

# correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

features = ['ParentalEducation', 'Tutoring', 'ParentalSupport', 'Extracurricular',
            'StudyTimeWeekly', 'Absences', 'Sports', 'Music', 'Volunteering']
target = 'GPA'

X = data[features]
y = data[target]

# preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['StudyTimeWeekly', 'Absences']),
        ('cat', OneHotEncoder(), ['ParentalEducation', 'Tutoring', 'ParentalSupport',
                                  'Extracurricular', 'Sports', 'Music', 'Volunteering'])
    ])

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a pipeline with a RandomForestRegressor
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# fit the model
rf_model.fit(X_train, y_train)

# cross-validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean CV score: {cv_scores.mean()}')

# evaluate on test set
test_score = rf_model.score(X_test, y_test)
print(f'Test Score: {test_score}')

# hyperparameter Tuning
# define parameter grid
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10]
}

# create a GridSearchCV object
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='r2', n_jobs=-1)

grid_search.fit(X_train, y_train)

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best cross-validation score: {grid_search.best_score_}')

# evaluate on test set
best_rf_model = grid_search.best_estimator_
test_score = best_rf_model.score(X_test, y_test)
print(f'Test Score: {test_score}')

# make predictions
y_pred = best_rf_model.predict(X_test)

# residual plot
plt.figure(figsize=(8, 6))
sns.residplot(x=y_test, y=y_pred, line_kws={'color': 'red', 'lw': 2})
plt.title('Residual Plot')
plt.xlabel('Actual GPA')
plt.ylabel('Residuals')
plt.show()

# predicted vs actual GPA plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.title('Predicted vs Actual GPA')
plt.xlabel('Actual GPA')
plt.ylabel('Predicted GPA')
plt.show()
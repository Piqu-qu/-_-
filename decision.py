import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Загрузка данных
data = pd.read_csv('C:\\Users\\User\\Downloads\\train_dataset_train_data_NPF\\train_data\\cntrbtrs_clnts_ops_trn.csv', sep=';', low_memory=False)

# Разносим полученные данные в разные переменные
y = data['erly_pnsn_flg']
X = data.drop(columns=['erly_pnsn_flg', 'accnt_bgn_date'])  # Удаляем целевую переменную и ненужный столбец

# Предварительная обработка данных
data['pstl_code'] = pd.to_numeric(data['pstl_code'], errors='coerce')

# Создаем модель
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)

# Прописываем кросс-валидацию
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f'Оценки кросс-валидации: {cv_scores}')
print(f'Средняя точность: {cv_scores.mean():.2f}')

# Обучение модели
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Выводим данные по полученной точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

proba_df = pd.DataFrame(y_proba, columns=model.classes_)
print("Вероятности для тестовых данных:")
print(proba_df)

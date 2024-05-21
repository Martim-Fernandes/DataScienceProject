import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

# Carregamento dos dados
caminho_do_arquivo = r'C:\Users\marti\Downloads\hcc_dataset.csv'
dados = pd.read_csv(caminho_do_arquivo, na_values='?', keep_default_na=False)

# Remoção de linhas com alta porcentagem de dados ausentes
percent_missing = dados.isna().mean(axis=1) * 100
indices_to_remove = dados.index[percent_missing > 20].tolist()
dados = dados.drop(indices_to_remove)

# Processamento de colunas numéricas
colunas_numericas = dados.select_dtypes(include=['number']).columns
dados[colunas_numericas] = dados[colunas_numericas].apply(pd.to_numeric, errors='coerce')

# Imputação de valores NaN
imputer = SimpleImputer(strategy='median')
dados[colunas_numericas] = imputer.fit_transform(dados[colunas_numericas])

# Identificação e remoção de colunas altamente correlacionadas
correlation_matrix = dados[colunas_numericas].corr()
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool_))
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.9)]
dados.drop(columns=to_drop, inplace=True)

# Aplicação de One-Hot Encoding em colunas categóricas, exceto 'Encefalopatia' e 'gender'
colunas_categoricas = [col for col in dados.columns if dados[col].dtype == 'object' and col not in ['Encefalopatia', 'gender']]
dados = pd.get_dummies(dados, columns=colunas_categoricas)

# Divisão dos dados em conjuntos de treinamento e teste
X = dados.drop(['Class_Dies', 'Class_Lives'], axis=1)
y = dados['Class_Dies']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criação de ficheiro com o novo conjunto de dados
dados.to_csv(r'C:\Users\marti\Downloads\hcc_dataset_cleaned.csv', index=False)

# Balanceamento de classes com SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Normalização das features numéricas
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# Definição e avaliação de múltiplos modelos
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(random_state=42)
}

# Definição e treinamento do modelo RandomForest
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train_res, y_train_res)

# Obtenção da importância das características
feature_importances_rf = random_forest_model.feature_importances_

# Criação de um DataFrame para visualização
features_df_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances_rf
}).sort_values(by='Importance', ascending=False)

# Importâncias das características para Random Forest
plt.figure(figsize=(12, 18))  
sns.barplot(x='Importance', y='Feature', data=features_df_rf)
plt.title('Importância das Características (Random Forest)')
plt.xlabel('Importância')
plt.ylabel('Característica')
plt.tight_layout()  
plt.show()

# Definição e treinamento do modelo Decision Tree
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train_res, y_train_res)

# Obtenção da importância das características
feature_importances_dt = decision_tree_model.feature_importances_

# Criação de um DataFrame para visualização
features_df_dt = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances_dt
}).sort_values(by='Importance', ascending=False)

# Importâncias das características para Decision Tree
plt.figure(figsize=(12, 18))  
sns.barplot(x='Importance', y='Feature', data=features_df_dt)
plt.title('Importância das Características (Decision Tree)')
plt.xlabel('Importância')
plt.ylabel('Característica')
plt.tight_layout()  
plt.show()

results = {}
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    results[name] = (accuracy, precision, recall, f1)

# Exibindo os resultados
for name, scores in results.items():
    print(f"{name} - Accuracy: {scores[0]:.2f}, Precision: {scores[1]:.2f}, Recall: {scores[2]:.2f}, F1 Score: {scores[3]:.2f}")

# Informações sobre os dados processados
if indices_to_remove:
    adjusted_indices = [index + 2 for index in indices_to_remove]
    adjusted_indices_person = [index + 1 for index in indices_to_remove]
    print("Linhas removidas:", adjusted_indices)
    print("Correspondentes às pessoas:", adjusted_indices_person)
else:
    print("Nenhuma linha foi removida.")

# Análise exploratória de dados após treinamento
# Primeiro, é necessário carregar ou definir as colunas 'Class_Dies' e 'Class_Lives' se ainda não estiverem no dataset
# Aqui, estou supondo que 'Class_Dies' é a variável 'y' e 'Class_Lives' é o inverso
class_counts = dados['Class_Dies'].value_counts().rename(index={0: 'Class_Lives', 1: 'Class_Dies'})

# Plotando a distribuição das classes alvo
plt.figure(figsize=(6, 4))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title('Sobrevivência dos pacientes')
plt.xlabel('Vive/Morre')
plt.ylabel('Contagem')
plt.show()

numerical_columns = dados.select_dtypes(include=['number']).columns
for coluna in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(dados[coluna], kde=True)
    plt.title(f'Distribuição de {coluna}')
    plt.show()

for coluna in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=dados[coluna])
    plt.title(f'Box Plot de {coluna}')
    plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(dados[numerical_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação')
plt.show()

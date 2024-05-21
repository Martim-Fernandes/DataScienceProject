# DataScienceProject

Guia de Uso e Execução do Programa

Introdução

Este guia tem como objetivo fornecer instruções detalhadas para a execução do programa que processa e analisa um conjunto de dados de um estudo sobre câncer hepático. O programa realiza diversas etapas, desde o pré-processamento dos dados até a aplicação de modelos de machine learning para prever a sobrevivência dos pacientes. Este documento também aborda as dependências necessárias e a documentação básica para a compreensão e modificação do código.

# Dependências de Pacote

Antes de executar o programa, certifique-se de que todas as dependências necessárias estão instaladas. Os pacotes utilizados são:

pandas

seaborn

matplotlib

numpy

scikit-learn

imbalanced-learn

Para instalar todas as dependências, você pode usar o seguinte comando:

pip install pandas seaborn matplotlib numpy scikit-learn imbalanced-learn


# Estrutura do Projeto

O projeto consiste em dois arquivos principais:

DSproject.py: Contém todo o código para processamento dos dados, treinamento dos modelos e visualização dos resultados.
README.md: Este guia com instruções e documentação.


# Instruções de Execução

Clone o repositório ou copie os arquivos para o seu ambiente local.
Abra o Visual Studio Code e navegue até o diretório onde os arquivos estão localizados.


# Modifique os Caminhos dos Arquivos:

No arquivo main.py, encontrará linhas que especificam o caminho do arquivo de dados. Modifique esses caminhos conforme necessário para corresponder à localização dos seus arquivos:
caminho_do_arquivo = r'C:\Users\marti\Downloads\hcc_dataset.csv'


# Ajuste o Limite de Correlação:

O valor limite para a remoção de colunas altamente correlacionadas está definido como 0.9. Você pode modificar este valor de acordo com suas necessidades:
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.9)]


# Execute o Programa:

Para executar o programa, abra o terminal integrado do Visual Studio Code e digite:
python DSproject.py


# Documentação Básica

# Carregamento dos Dados

O programa começa por carregar os dados de um arquivo CSV especificado. Certifique-se de que o caminho do arquivo está correto.
caminho_do_arquivo = r'C:\Users\marti\Downloads\hcc_dataset.csv'
dados = pd.read_csv(caminho_do_arquivo, na_values='?', keep_default_na=False)

# Pré-Processamento

Remoção de Linhas com Alta Percentagem de Dados Ausentes:
Linhas com mais de 20% de valores ausentes são removidas, mas se pretender pode modificar essa valor.
percent_missing = dados.isna().mean(axis=1) * 100
indices_to_remove = dados.index[percent_missing > 20].tolist()
dados = dados.drop(indices_to_remove)

# Conversão e Imputação de Valores Numéricos:

Colunas numéricas são convertidas para o tipo correto e valores ausentes são preenchidos com a mediana.
colunas_numericas = dados.select_dtypes(include=['number']).columns
dados[colunas_numericas] = dados[colunas_numericas].apply(pd.to_numeric, errors='coerce')
imputer = SimpleImputer(strategy='median')
dados[colunas_numericas] = imputer.fit_transform(dados[colunas_numericas])

# Remoção de Colunas Altamente Correlacionadas:

Colunas com correlação acima de um valor limite (0.9 por padrão) são removidas.
correlation_matrix = dados[colunas_numericas].corr()
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool_))
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.9)]
dados.drop(columns=to_drop, inplace=True)

# Codificação de Variáveis Categóricas:

Variáveis categóricas são convertidas usando One-Hot Encoding, exceto as colunas 'Encefalopatia' e 'gender'.
colunas_categoricas = [col for col in dados.columns if dados[col].dtype == 'object' and col not in ['Encefalopatia', 'gender']]
dados = pd.get_dummies(dados, columns=colunas_categoricas)

# Divisão e Balanceamento dos Dados

Os dados são divididos em conjuntos de treinamento e teste, e o conjunto de treinamento é balanceado usando SMOTE.
X = dados.drop(['Class_Dies', 'Class_Lives'], axis=1)
y = dados['Class_Dies']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# Treinamento e Avaliação de Modelos

Modelos de machine learning são treinados e avaliados. Os resultados são exibidos para cada modelo.
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    results[name] = (accuracy, precision, recall, f1)

for name, scores in results.items():
    print(f"{name} - Accuracy: {scores[0]:.2f}, Precision: {scores[1]:.2f}, Recall: {scores[2]:.2f}, F1 Score: {scores[3]:.2f}")
    
Visualização dos Resultados
Diversas visualizações são geradas para ajudar na análise dos dados e dos resultados dos modelos.

# Importância das características
plt.figure(figsize=(12, 18))  
sns.barplot(x='Importance', y='Feature', data=features_df_rf)
plt.title('Importância das Características (Random Forest)')
plt.xlabel('Importância')
plt.ylabel('Característica')
plt.tight_layout()  
plt.show()

# Contagem das classes de sobrevivência
class_counts = dados['Class_Dies'].value_counts().rename(index={0: 'Class_Lives', 1: 'Class_Dies'})
plt.figure(figsize=(6, 4))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title('Sobrevivência dos pacientes')
plt.xlabel('Vive/Morre')
plt.ylabel('Contagem')
plt.show()

# Histogramas e boxplots das colunas numéricas
for coluna in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(dados[coluna], kde=True)
    plt.title(f'Distribuição de {coluna}')
    plt.show()
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=dados[coluna])
    plt.title(f'Box Plot de {coluna}')
    plt.show()

# Matriz de correlação
plt.figure(figsize=(12, 10))
sns.heatmap(dados[numerical_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação')
plt.show()

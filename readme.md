# Open Food Chat 🍔
ChatBot educacional para responder perguntas básicas sobre alimentos usando a base de dados Open Food Facts.

# Docs 📚

- [PMC](./docs/pmc.md)
- [Arquitetura](./docs/architeture.md)
- [DataModel](./docs/datamodel.md)

# 
# Como rodar o Projeto? 🛠️

### 1. Clone o repositório
```
git clone https://github.com/ewertonlx/ia.git
```
### 2. Vá até o diretório
```
cd ia
```
### 3. Abra no VSCode ou no seu Editor de Texto
```
vscode: code .
// Execute isso dentro do diretório do repositório
```
### 4. Criar e ativar o ambiente virtual
```
python -m venv .venv

# Ativar no Linux/Mac
source .venv/bin/activate

# Ativar no Windows (PowerShell)
.venv\Scripts\Activate.ps1
```
### 5. Instalar as dependências do projeto
```
pip install -r requirements.txt
```

### 6. Rodar o Streamlit
```
streamlit run app/app.py
```

- Com isso ele irá abrir uma página no seu navegador em localhost.

##
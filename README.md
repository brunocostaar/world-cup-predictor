# FIFA World Cup 2026 Predictor & Simulator 🏆⚽

Este projeto é um simulador e preditor de partidas para a **Copa do Mundo FIFA de 2026** (com o novo formato de 48 seleções). Ele combina raspagem de dados reais de futebol, modelos de Machine Learning (Random Forest e Regressão de Poisson regularizada) e simulações estocásticas de Monte Carlo para gerar estatísticas probabilísticas refinadas e uma interface de usuário rica no Streamlit.

---

## 🚀 Como Mudar o Número de Simulações

### 1. No Terminal (via script `main.py`)
Abra o arquivo [main.py](file:///home/bruno/AntiGravity/predictor/main.py) e altere o valor da variável `sim_runs` na linha 22:
```python
# Altere de 1000 para o número desejado, ex: 100 para testes rápidos ou 10000 para precisão máxima
sim_runs = 100 
```
Após salvar o arquivo, execute no terminal:
```bash
python3 main.py
```

### 2. Na Interface Gráfica (Streamlit)
Você não precisa mexer no código! Na barra lateral esquerda do painel Streamlit, existe um controle deslizante chamado **"Número de Simulações"**.
* Basta mover o **slider** para selecionar qualquer valor entre **100 e 2.000** simulações.
* O simulador de Monte Carlo recalculará automaticamente e atualizará todas as tabelas e gráficos em tempo real.

---

## 📁 Estrutura do Workspace

O projeto está dividido em módulos funcionais e independentes:

```
predictor/
├── venv/                            # Ambiente virtual Python
├── .streamlit/
│   └── config.toml                  # Configuração do tema escuro premium
├── data/                            # Datasets, arquivos sementes e modelos pickle
│   ├── world_cup_2026_groups.json   # Composição oficial dos grupos A a L
│   ├── fifa_rankings_seed.csv       # Fallback de rankings da FIFA
│   ├── squad_values_seed.csv        # Fallback de valores de mercado
│   ├── team_metadata_seed.csv       # FC 26 Ratings e PPG das Eliminatórias (Semente)
│   ├── historical_matches.csv       # Histórico de partidas reais para treino da IA
│   ├── final_features.csv           # Base alinhada e calibrada de 48 seleções
│   ├── rf_model.pkl                 # Modelo Random Forest treinado
│   └── poisson_model.pkl            # Modelo de Gols Simétrico (Poisson) treinado
├── fetchers/                        # Módulos de raspagem de dados (FIFA, Transfermarkt, Forma)
│   ├── fifa_ranking.py
│   ├── transfermarkt.py
│   └── recent_form.py
├── processing/
│   └── prepare_dataset.py           # Normalização de nomes e pipeline de consolidação
├── model/
│   ├── train.py                     # Treinamento dos modelos de Machine Learning
│   └── predict.py                   # Mecanismo de simulação estocástica da Copa
├── app.py                           # Dashboard Streamlit interativo
├── main.py                          # Pipeline orquestrador console
├── requirements.txt                 # Dependências do Python
├── test_data_gathering.py           # Testes unitários do pipeline de dados
└── test_predictions.py              # Testes unitários de previsão e chaves do mata-mata
```

---

## 🧠 Como Funciona o Modelo de IA?

O simulador utiliza um pipeline híbrido de Machine Learning:

### 1. Acurácia e Importância dos Atributos (Random Forest)
O classificador de Floresta Aleatória treina no histórico de confrontos ([historical_matches.csv](file:///home/bruno/AntiGravity/predictor/data/historical_matches.csv)) para avaliar a influência de cada variável na probabilidade de vitória/empate/derrota:
* **Diferença de Rankings FIFA (`rank_diff`)**: ~23.9%
* **Diferença de Forma Recente (`form_diff`)**: ~23.2%
* **PPG nas Eliminatórias da Copa (`qualifiers_ppg_diff`)**: ~18.6%
* **Valor Logarítmico de Elenco (`log_value_ratio`)**: ~17.9%
* **Diferença de Atributo de Cartas FC26 (`fc26_diff`)**: ~16.4%

### 2. Gols Marcados (Regressão de Poisson Simétrica e Regularizada)
Para partidas em campos neutros no Mundial, usamos um modelo único e simétrico de Poisson (`poisson_model.pkl`):
* **Simetria**: O modelo é treinado espelhando os jogos de forma que os gols de um time sejam previstos puramente a partir da diferença de atributos em relação ao seu oponente ($Team_A - Team_B$). Isso elimina o "efeito mandante" inerente aos confrontos.
* **Regularização L2 ($\alpha = 0.1$)**: Pequenas oscilações em dados de forma recente (como amistosos de fim de ano) são atenuadas para impedir que distorçam as previsões de longo prazo baseadas em qualidade geral e ranking.

---

## ⚖️ Regras de Calibração Esportiva (Correção de Vieses)

Para evitar distorções de dados brutos e mercadológicos, o sistema aplica calibrações em [config.py](file:///home/bruno/AntiGravity/predictor/config.py):
1. **Multiplicadores de Confederação**: O PPG e Rankings da FIFA são ajustados pela dificuldade de calendário de sua confederação regional (UEFA: 1.0, CONMEBOL: 0.96, CAF: 0.82, Concacaf: 0.80, AFC: 0.76, OFC: 0.65). Isso impede que seleções asiáticas ou africanas tenham métricas infladas por golearem adversários locais mais fracos.
2. **Desconto de Inflação Inglês**: Aplica-se um desconto de **25%** (`0.75`) no valor de mercado do elenco da Inglaterra para neutralizar a bolha financeira de jogadores "homegrown" da Premier League.

---

## 💻 Instalação e Execução

### Pré-requisitos
Certifique-se de ter o Python 3.10+ instalado no seu sistema Linux/macOS.

### 1. Configurar o Ambiente e Dependências
Abra o terminal no diretório do projeto e execute:
```bash
# Ativar o ambiente virtual já configurado
source venv/bin/activate

# Instalar/Atualizar as dependências
pip install -r requirements.txt
```

### 2. Executar o Orquestrador Console (Treinar e Simular)
Para regenerar os dados de mercado, rankings, retreinar as IAs e simular o torneio gerando tabelas de classificação no terminal:
```bash
python3 main.py
```

### 3. Executar Testes Unitários
Para verificar a consistência e integridade das funções e regras matemáticas de simulação:
```bash
python3 -m unittest discover -s . -p "test_*.py"
```

### 4. Abrir a Interface Streamlit (Web App)
Para carregar o dashboard interativo moderno:
```bash
streamlit run app.py
```
O Streamlit abrirá uma nova aba automaticamente no seu navegador padrão em **http://localhost:8501**.

---

## 📊 Funcionalidades do Dashboard Web
* **🏆 Favoritos & Grupos**: Distribuição de chances de classificação de cada grupo (A a L) e gráfico com as maiores probabilidades ao título mundial.
* **🌿 Simulador do Torneio**: Simule a Copa inteira na hora! Veja os classificados, os 8 melhores terceiros colocados e a árvore do mata-mata dinâmica (dos 32 avos até a grande final).
* **⚔️ Simulador de Confrontos**: Monte confrontos personalizados entre duas seleções e veja as chances de vitória, empates e as estatísticas dos placares mais prováveis.
* **📁 Banco de Dados**: Painel para visualização e busca das estatísticas brutas normalizadas de todas as 48 seleções participantes.

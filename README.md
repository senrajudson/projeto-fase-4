## Projeto Fase 4 - Pipeline de Treinamento e API

### Parte 1 - Explicação do projeto
Este projeto tem duas entregas principais: uma pipeline de treinamento de modelo LSTM para previsao de series temporais e uma API FastAPI para servir o modelo treinado.
Na pipeline, os dados sao coletados do Yahoo Finance, passam por limpeza e normalizacao, e geram checkpoints do modelo. A API carrega um desses checkpoints e expoe um endpoint de previsao.

### Parte 2 - Explicacao do codigo
#### Estrutura de pastas
- pipeline: pipeline de treinamento e scripts de otimizacao com Optuna.
- api: servico FastAPI para inferencia do modelo.
- data: dados locais (se usados/gerados).
- checkpoints: checkpoints exportados para a API (montado no docker-compose).

#### Pipeline de treinamento (pipeline/)
- Entrada principal: `pipeline/main_train.py`.
- Fluxo:
  1) `pipeline/data/1_source_yahoo.py` baixa a série de preços (yfinance).
  2) `pipeline/preprocessing/1_integrity.py` valida e limpa valores inválidos.
  3) `pipeline/preprocessing/2_normalization.py` calcula média/desvio e normaliza.
  4) `pipeline/training/1_train.py` treina o LSTM e salva checkpoint.
- Configuração central: `pipeline/config.py` (classe `TrainConfig`).

#### Métricas (MAE, RMSE, MAPE) e quando são calculadas
- As métricas são calculadas no final do treino, depois que o modelo carrega o melhor estado (menor perda de validação).
- O cálculo acontece em `pipeline/training/2_evaluate.py` dentro de `evaluate_denorm_metrics`.
- Passo a passo:
  1) O treino roda com MSE na validação e guarda o melhor estado.
  2) Ao final, o modelo carrega o melhor estado.
  3) No conjunto de teste, as previsões e os alvos são desnormalizados (usando média e desvio do treino).
  4) As métricas são computadas:
     - MAE = média do erro absoluto.
     - RMSE = raiz da média do erro quadrático.
     - MAPE = média do erro percentual absoluto (com clamp para evitar divisão por zero).
- As métricas ficam registradas no checkpoint em `metrics_test` e também aparecem no retorno de `pipeline/main_train.py`.

#### Otimização com Optuna (pipeline/)
- Entrada principal: `pipeline/main_optuna.py`.
- A função `run_optuna_with_series` em `pipeline/training/4_optuna_hpo.py` executa a busca.
- Os resultados vao para um SQLite definido em `TrainConfig.storage_path`.
- O melhor checkpoint pode ser exportado como `best_pareto_lstm.pt`.

#### Optuna passo a passo
1) A série é coletada e limpa (mesmo fluxo do treino).
2) A série é normalizada usando média/desvio do treino.
3) O estudo e criado com tres objetivos: minimizar MAE, RMSE e MAPE.
4) Cada trial ajusta hiperparâmetros (sequence_length, hidden_size, num_layers, dropout, learning_rate, batch_size, max_epochs, patience).
5) O treino roda, gera checkpoint por trial e calcula metricas no conjunto de teste.
6) O Optuna registra MAE, RMSE e MAPE como objetivos do trial.
7) O pareto e exportado em CSV.
8) O melhor trial é escolhido pela estratégia (`min_mape`, `min_mae`, `min_rmse` ou `weighted`).
9) O checkpoint do melhor trial é salvo como `best_checkpoint_path`.

#### Inferência local (pipeline/)
- Entrada principal: `pipeline/main_infer.py`.
- Carrega um checkpoint e executa `predict_next_from_values` com uma lista de floats.

#### API (api/)
- Entrada principal: `api/src/main_api.py`.
- Endpoint: `POST /predict`
  - body: `{ "values": [1.0, 2.0, ...] }`
  - resposta: previsao e metadados do checkpoint.
- Modelo de inferência (singleton): `api/src/model/inference.py`.
  - O modelo e carregado uma unica vez e reutilizado em todas as requisicoes.
  - Usa `cuda` se disponivel, caso contrario usa `cpu`.

#### API - Requisição e saída
Requisicao (JSON)
```json
{
  "values": [120.5, 121.0, 122.3, 121.8, 123.1]
}
```
Saida (JSON)
```json
{
  "ok": true,
  "prediction": 123.9,
  "checkpoint_path": "checkpoints/best.pt",
  "sequence_length": 60,
  "symbol": "BTC-USD",
  "feature": "Close"
}
```
Observações
- `values` deve conter uma lista de floats com tamanho igual ao `sequence_length` usado no treino.
- O checkpoint e fixo em `/app/model/best_pareto_lstm.pt` e deve existir no container.

#### Docker
- `docker-compose.yaml` sobe a API e monta `./api/modelo` em `/app/model`.
- O container expõe a porta 9010.

### Parte 3 - Como executar o código
#### Requisitos
- Python 3.11+
- Dependências em `api/pyproject.toml`

#### Tutorial completo - Pipeline (venv + dependências)
1) Entre na pasta do pipeline:
```bash
cd pipeline
```
2) Remova qualquer ambiente antigo:
```bash
deactivate 2>/dev/null || true
rm -rf .venv venv
```
3) Crie um novo ambiente virtual:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
4) Atualize ferramentas base:
```bash
python -m pip install -U pip setuptools wheel
```
5) Instale as dependências:
```bash
pip install --no-cache-dir -r requirements.txt
```
6) Verifique se o ambiente está saudável:
```bash
pip check
```
Se não aparecer nenhum erro, o ambiente está pronto.

7) Execute o treinamento:
```bash
python main_train.py
```
8) (Opcional) Rode Optuna:
```bash
python main_optuna.py
```
9) (Opcional) Rode a inferência local:
```bash
python main_infer.py
```

Observação:
- Pode ocorrer erro relacionado a CUDA dependendo da versão da sua placa de vídeo e do build do PyTorch instalado.

#### Dashboard do Optuna
1) Instale o dashboard do Optuna (se necessário):
```bash
pip install optuna-dashboard
```
2) Inicie o dashboard apontando para o banco do estudo:
```bash
optuna-dashboard sqlite:///optuna_study.db --host 127.0.0.1 --port 8080
```
3) Acesse no navegador: `http://127.0.0.1:8080`

#### Tutorial completo - API com Docker Compose
1) Garanta que exista um checkpoint em `checkpoints/`.
2) Suba a API com Docker:
```bash
docker compose up --build
```
3) A API estará disponível em `http://localhost:9010`.

## Projeto Fase 4 - Pipeline de Treinamento e API

### Parte 1 - Explicacao do projeto
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
  1) `pipeline/data/1_source_yahoo.py` baixa a serie de precos (yfinance).
  2) `pipeline/preprocessing/1_integrity.py` valida e limpa valores invalidos.
  3) `pipeline/preprocessing/2_normalization.py` calcula media/desvio e normaliza.
  4) `pipeline/training/1_train.py` treina o LSTM e salva checkpoint.
- Configuracao central: `pipeline/config.py` (classe `TrainConfig`).

#### Metricas (MAE, RMSE, MAPE) e quando sao calculadas
- As metricas sao calculadas no final do treino, depois que o modelo carrega o melhor estado (menor perda de validacao).
- O calculo acontece em `pipeline/training/2_evaluate.py` dentro de `evaluate_denorm_metrics`.
- Passo a passo:
  1) O treino roda com MSE na validacao e guarda o melhor estado.
  2) Ao final, o modelo carrega o melhor estado.
  3) No conjunto de teste, as previsoes e os alvos sao desnormalizados (usando media e desvio do treino).
  4) As metricas sao computadas:
     - MAE = media do erro absoluto.
     - RMSE = raiz da media do erro quadratico.
     - MAPE = media do erro percentual absoluto (com clamp para evitar divisao por zero).
- As metricas ficam registradas no checkpoint em `metrics_test` e tambem aparecem no retorno de `pipeline/main_train.py`.

#### Otimizacao com Optuna (pipeline/)
- Entrada principal: `pipeline/main_optuna.py`.
- A funcao `run_optuna_with_series` em `pipeline/training/4_optuna_hpo.py` executa a busca.
- Os resultados vao para um SQLite definido em `TrainConfig.storage_path`.
- O melhor checkpoint pode ser exportado como `best_pareto_lstm.pt`.

#### Optuna passo a passo
1) A serie e coletada e limpa (mesmo fluxo do treino).
2) A serie e normalizada usando media/desvio do treino.
3) O estudo e criado com tres objetivos: minimizar MAE, RMSE e MAPE.
4) Cada trial ajusta hiperparametros (sequence_length, hidden_size, num_layers, dropout, learning_rate, batch_size, max_epochs, patience).
5) O treino roda, gera checkpoint por trial e calcula metricas no conjunto de teste.
6) O Optuna registra MAE, RMSE e MAPE como objetivos do trial.
7) O pareto e exportado em CSV.
8) O melhor trial e escolhido pela estrategia (`min_mape`, `min_mae`, `min_rmse` ou `weighted`).
9) O checkpoint do melhor trial e salvo como `best_checkpoint_path`.

#### Inferencia local (pipeline/)
- Entrada principal: `pipeline/main_infer.py`.
- Carrega um checkpoint e executa `predict_next_from_values` com uma lista de floats.

#### API (api/)
- Entrada principal: `api/src/main_api.py`.
- Endpoint: `POST /predict`
  - body: `{ "values": [1.0, 2.0, ...], "checkpoint_path": "checkpoints/best.pt", "device": "cpu" }`
  - resposta: previsao e metadados do checkpoint.
- Modelo de inferencia: `api/src/model/main_infer.py`.

#### API - Requisicao e saida
Requisicao (JSON)
```json
{
  "values": [120.5, 121.0, 122.3, 121.8, 123.1],
  "checkpoint_path": "checkpoints/best.pt",
  "device": "cpu"
}
```
Saida (JSON)
```json
{
  "prediction": 123.9,
  "checkpoint_path": "checkpoints/best.pt",
  "device": "cpu",
  "metrics": {
    "MAE": 0.0,
    "RMSE": 0.0,
    "MAPE": 0.0
  }
}
```
Observacoes
- `values` deve conter uma lista de floats com tamanho igual ao `sequence_length` usado no treino.
- `checkpoint_path` aponta para o arquivo do checkpoint montado no container.

#### Docker
- `docker-compose.yaml` sobe a API e monta `./checkpoints` em `/app/checkpoints`.
- O container expoe a porta 9010.

### Parte 3 - Como executar o codigo
#### Requisitos
- Python 3.11+
- Dependencias em `api/pyproject.toml`

#### Tutorial completo - Pipeline (venv + dependencias)
1) Crie a venv na raiz do projeto:
```bash
python -m venv .venv
```
2) Ative a venv:
```bash
source .venv/bin/activate
```
3) Atualize o pip e instale as dependencias:
```bash
python -m pip install --upgrade pip
python -m pip install -r pipeline/requirements.txt
python -m pip install -e api
```
4) Execute a pipeline de treinamento:
```bash
python -m pipeline.main_train
```
5) (Opcional) Rode Optuna:
```bash
python -m pipeline.main_optuna
```
6) (Opcional) Rode a inferencia local:
```bash
python -m pipeline.main_infer
```

#### Tutorial completo - API com Docker Compose
1) Garanta que exista um checkpoint em `checkpoints/`.
2) Suba a API com Docker:
```bash
docker compose up --build
```
3) A API estara disponivel em `http://localhost:9010`.

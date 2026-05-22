# Elasticsearch Backend

Pipeline 1 can use Elasticsearch as a dense-vector backend in two modes:

- `script_score`: exact dense retrieval using `cosineSimilarity`. This is the default fallback and preserves older Elasticsearch configs.
- `knn`: native Elasticsearch approximate nearest-neighbor retrieval over indexed `dense_vector` fields.

## Start Elasticsearch

For local unauthenticated validation:

```bash
docker run --rm --name rag-es \
  -p 9200:9200 \
  -e discovery.type=single-node \
  -e xpack.security.enabled=false \
  -e ES_JAVA_OPTS="-Xms1g -Xmx1g" \
  docker.elastic.co/elasticsearch/elasticsearch:8.13.4
```

Wait until the cluster responds:

```bash
curl http://localhost:9200
```

## Smoke Validation

The smoke script creates a temporary synthetic index, indexes four known vectors, runs retrieval, verifies the expected top hit, and deletes the index unless `--keep-index` is set.

```bash
python scripts/smoke_elasticsearch_backend.py --host http://localhost:9200 --mode script_score
python scripts/smoke_elasticsearch_backend.py --host http://localhost:9200 --mode knn
python scripts/smoke_elasticsearch_backend.py --host http://localhost:9200 --mode both
```

With auth:

```bash
python scripts/smoke_elasticsearch_backend.py \
  --host https://localhost:9200 \
  --username elastic \
  --password "$ELASTIC_PASSWORD" \
  --verify-certs \
  --mode both
```

The script also reads `ELASTICSEARCH_HOST`, `ELASTICSEARCH_USERNAME`, `ELASTICSEARCH_PASSWORD`, `ELASTICSEARCH_API_KEY`, `ELASTICSEARCH_DENSE_DIM`, and `ELASTICSEARCH_SMOKE_INDEX`.

## Run Pipeline 1

Script-score Elasticsearch:

```bash
python -m src.pipeline1.main --config configs/pipeline1/experiments/officeqa_treasury_fixed512_bge_small_elastic_script_score_qwen25_7b_ctxbudget.yaml
```

Native kNN Elasticsearch:

```bash
python -m src.pipeline1.main --config configs/pipeline1/experiments/officeqa_treasury_fixed512_bge_small_elastic_knn_qwen25_7b_ctxbudget.yaml
```

Both configs mirror the fixed512 BGE-small FAISS baseline and only swap the index and retriever to Elasticsearch.

## Run Pipeline 2

After Pipeline 1 writes `results.jsonl`, evaluate the script-score run:

```bash
python -m src.pipeline2.main --config configs/pipeline2/experiments/eval_officeqa_treasury_fixed512_bge_small_elastic_script_score_qwen25_7b_fileeval_ks1_3_5_10_numacc.yaml
```

Evaluate the kNN run:

```bash
python -m src.pipeline2.main --config configs/pipeline2/experiments/eval_officeqa_treasury_fixed512_bge_small_elastic_knn_qwen25_7b_fileeval_ks1_3_5_10_numacc.yaml
```

## Compare Against FAISS

Run the matching FAISS baseline first:

```bash
python -m src.pipeline1.main --config configs/pipeline1/experiments/03_officeqa_treasury_fixed512_bge_small_dense_norerank_fetch50_qwen25_7b_ctxbudget.yaml
python -m src.pipeline2.main --config configs/pipeline2/experiments/03_eval_officeqa_treasury_fixed512_bge_small_dense_norerank_fetch50_qwen25_7b_fileeval_ks1_3_5_10_numacc.yaml
```

Then inspect the Pipeline 2 summaries:

```bash
python scripts/compare_runs.py \
  data/eval/runs/pipeline2/03_eval_officeqa_treasury_fixed512_bge_small_dense_norerank_fetch50_qwen25_7b_fileeval_ks1_3_5_10_numacc/summary_by_experiment.csv \
  data/eval/runs/pipeline2/eval_officeqa_treasury_fixed512_bge_small_elastic_script_score_qwen25_7b_fileeval_ks1_3_5_10_numacc/summary_by_experiment.csv \
  data/eval/runs/pipeline2/eval_officeqa_treasury_fixed512_bge_small_elastic_knn_qwen25_7b_fileeval_ks1_3_5_10_numacc/summary_by_experiment.csv
```

## Warnings

Elasticsearch and FAISS scores are not guaranteed to be numerically comparable. Compare rankings and Pipeline 2 metrics, not raw score magnitudes.

`script_score` is exact but can be slow because it scores candidate documents directly. It is useful as a correctness baseline and for small corpora.

Native `knn` is approximate. It can be faster at scale, but results may differ from exact FAISS/script-score retrieval. Increase `index.num_candidates` if recall is too low, then measure the latency and quality tradeoff in Pipeline 2.

The local Docker command disables security for developer validation only. Do not use that setup for shared or production clusters.

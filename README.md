
# ARR-LHRM
## What is ARR-LHRM?

**ARR-LHRM** combines:
- **Memory internalization** (Self-Retrieval-style): internalize a corpus into model parameters and retrieve by generation + self-assessment.
- **Proactive hybrid reasoning** (Think-only-when-needed): decide whether to trigger external search only when internal evidence is low-confidence.

This repo is an ARR-LHRM implementation built on top of the Self-Retrieval idea. See the citation section for the original Self-Retrieval paper.

For more information, checkout our [paper](https://arxiv.org/pdf/2403.00801).

## Data Preprocess
- download the original data from [google drive](https://drive.google.com/drive/folders/1GfW0WxQUTnAz0pJ5WRyVFgUhni-yz75f?usp=sharing).
- build indexing data

```bash
export data_dir=../data/NQ
export index_model_path=../models/NQ/index_llama
export model_path=../models/NQ/arr_lhrm_llama
export result_dir=../retrieve_outputs/NQ/arr_lhrm_llama
```

```bash
cd data_preprocess

python mem_dataset.py \
    --data_path ${data_dir}/docs.json \
    --output_path ${data_dir}/mem_all.json \
    --use_all_sents \
    --lower_title \
    --tile_and_prefix_only

python mem_dataset.py \
    --data_path ${data_dir}/docs.json \
    --output_path ${data_dir}/mem_3.json \
    --random_sent_num \
    --lower_title \
    --tile_and_prefix_only
```

- add negative data to the training set
```bash
python add_neg_data.py \
    --data_path ${data_dir}/train.json \
    --output_path ${data_dir}/train_neg5.json \
    --doc_path ${data_dir}/docs.json \
    --neg_num 5
```

- build the training dataset
```bash
python qa_dataset.py \
    --qa_path ${data_dir}/train_neg5.json \
    --doc_path ${data_dir}/docs.json \
    --mem_data_path ${data_dir}/mem_3.json \
    --output_path ${data_dir}/mem3_neg5_cfg1_repeat2_llama2.json \
    --config_path ./configs/config1.json \
    --qa_repeat_times 2 \
    --eos_token '</s>'
```
## Training

After completing data preprocessing, you can use the scripts `sft.sh` for model training.

### Hardware / Environment
- **GPU**: NVIDIA GeForce **RTX 5090** (default examples assume **single-GPU** training)
- **OS**: Linux / WSL2 is recommended for training (Deepspeed)
- **Multi-GPU**: set `NUM_GPUS` to match your available GPUs

```bash
cd train

NUM_GPUS=1 bash sft.sh ${data_dir}/mem_all.json meta-llama/Llama-2-7b-hf ${index_model_path}

NUM_GPUS=1 bash sft.sh ${data_dir}/mem3_neg5_cfg1_repeat2_llama2.json ${index_model_path} ${model_path}
```

## ARR-LHRM: 3-Mode Routing (Direct / Retrieve / Reason)
This repo includes a doc-aligned **three-mode** inference/training scaffold:
- **Direct**: answer directly (`<no_think>`)
- **Retrieve**: internal memory retrieval first, then answer with evidence (`<no_think>` + context)
- **Reason**: explicit multi-step reasoning (`<think> ... </think>`)

### Inference (ARR-LHRM)
```bash
python run.py arr \
  --query "Explain the basic principle of quantum entanglement." \
  --gen_model_path ${model_path} \
  --retrieval_model_path ${model_path} \
  --trie ${data_dir}/trie.json \
  --corpus ${data_dir}/docs.json
```

If you have a trained router:
```bash
python run.py arr \
  --query "What are the latest quantum entanglement experiments in 2024?" \
  --gen_model_path ${model_path} \
  --retrieval_model_path ${model_path} \
  --trie ${data_dir}/trie.json \
  --corpus ${data_dir}/docs.json \
  --router_path ../models/router_arr_lhrm_hgpo
```

### Stage I: Router HFT (Cold Start, supervised)
1) (Optional) Build an HFT dataset for the LLM SFT format:
```bash
cd data_preprocess
python arr_lhrm_hft_dataset.py \
  --qa_path ${data_dir}/train.json \
  --doc_path ${data_dir}/docs.json \
  --output_path ${data_dir}/arr_lhrm_hft.json \
  --router_labels_output ${data_dir}/router_labels.jsonl
```

2) Train router (supervised):
```bash
cd ../train
python router_cli.py hft \
  --base_model ${model_path} \
  --train_jsonl ${data_dir}/router_labels.jsonl \
  --output_dir ../models/router_arr_lhrm_hft
```

### Stage II: HGPO-style RL for router (lightweight, runnable)
Prepare a jsonl file with at least `query`, optional `gold` for accuracy reward:
```json
{"query": "2+2=?", "gold": "4"}
```

Then run:
```bash
python router_cli.py hgpo \
  --router_path ../models/router_arr_lhrm_hft \
  --gen_model ${model_path} \
  --train_jsonl ${data_dir}/router_rl.jsonl \
  --out_router_path ../models/router_arr_lhrm_hgpo \
  --trie ${data_dir}/trie.json \
  --corpus ${data_dir}/docs.json \
  --retrieval_model ${model_path}
```

> Note: Stage II here updates the **router policy πϕ** (mode selection) using a HGPO-style group comparison.
> Full token-level HGPO on the entire LLM (as in Hybrid-Reasoning) is not implemented in this repo.
## Evaluation
### Building the Tire
```bash
cd evaluation

python gettrie.py \
    --model_path meta-llama/Llama-2-7b-hf \
    --corpus ${data_dir}/docs.json \
    --outfile ${data_dir}/trie.json
```

### Evaluation for Retrieval
- predict
```bash
python predict.py \
    --model_path ${model_path} \
    --trie ${data_dir}/trie.json \
    --corpus ${data_dir}/docs.json \
    --data_path ${data_dir}/test.json \
    --output_dir ${result_dir} \
    --beam_num_for_title 5 \
    --beam_num_for_chunk 10
```
- get the final results
```bash
python get_retrieve_from_generation.py \
    --output_path ${result_dir} \
    --corpus ${data_dir}/docs.json \
    --data_path ${data_dir}/test.json
```

### Evaluation for RAG
```bash
python rag.py \
    --plm ${model_path} \
    --rplm ${model_path} \
    --lower True \
    --testsetfile ${result_dir}/retrieve.json \
    --contextfile ${data_dir}/docs.json
```

## Hybrid Reasoning (Proactive) + ARR-LHRM
If you want to combine **Hybrid-Reasoning** "Think only when you need" (i.e., decide when to spend extra cost)
with ARR-LHRM's **memory internalization**, we provide a lightweight two-stage orchestration:

- Stage 1 (**Internal / ARR-LHRM**): run constrained generation + self-assessment and compute a confidence score.
- Stage 2 (**Decision / Think**): if confidence is low (or uncertain), optionally try a local knowledge base fallback,
  and finally trigger an external web search client.

This is implemented in:
- `arr_lhrm/core.py` (two-stage orchestration + core logic)
- `run.py hybrid` (CLI)

### Quick start
1) Build trie (same as above):
```bash
cd evaluation
python gettrie.py --model_path meta-llama/Llama-2-7b-hf --corpus ${data_dir}/docs.json --outfile ${data_dir}/trie.json
```

2) Run hybrid pipeline on a single query (offline mode; no external search):
```bash
cd ..
python run.py hybrid \
  --query "Explain the basic principle of quantum entanglement." \
  --model_path ${model_path} \
  --trie ${data_dir}/trie.json \
  --corpus ${data_dir}/docs.json \
  --high_threshold 0.8 \
  --low_threshold 0.3
```

3) Enable local knowledge base fallback (cheap internal retrieval before web search):
```bash
python run.py hybrid \
  --query "What are the latest quantum entanglement experiments in 2024?" \
  --model_path ${model_path} \
  --trie ${data_dir}/trie.json \
  --corpus ${data_dir}/docs.json \
  --enable_local_kb \
  --uncertain_triggers_external
```

4) Enable external search:
- `--external_provider bing` requires `BING_SEARCH_KEY` (and optionally `BING_SEARCH_ENDPOINT`).

## Citation
If you find this work useful for your research, please cite:

```
@misc{tang2024selfretrievalendtoendinformationretrieval,
      title={Self-Retrieval: End-to-End Information Retrieval with One Large Language Model}, 
      author={Qiaoyu Tang and Jiawei Chen and Zhuoqun Li and Bowen Yu and Yaojie Lu and Cheng Fu and Haiyang Yu and Hongyu Lin and Fei Huang and Ben He and Xianpei Han and Le Sun and Yongbin Li},
      year={2024},
      eprint={2403.00801},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2403.00801}, 
}
```

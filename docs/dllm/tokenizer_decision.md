# Tokenizer — Decizie Finală

Data deciziei: 2026-04-09

## Tokenizer ales

```
HuggingFaceTB/SmolLM2-135M
vocab_size: 49152
algoritm: BPE (byte-level, tiktoken-style)
```

Aceasta este o decizie ireversibilă în contextul proiectului. Odată ce
antrenamentul de pretraining a început, tokenizer-ul nu se mai schimbă fără
a relua de la zero.

## Motivele deciziei

### 1. Dimensiunea vocab este optimă pentru 0.8B parametri

Un embedding table de dimensiune `V × D` contribuie direct la numărul total
de parametri. Cu `embedding_dim=1536` al modelului nostru:

| Tokenizer              | Vocab   | Embedding params | % din ~840M total |
| ---------------------- | ------- | ---------------- | ----------------- |
| Qwen/Qwen2.5-0.5B     | 151,936 | ~233M            | ~27%              |
| meta-llama/Llama-3.2   | 128,256 | ~197M            | ~23%              |
| **SmolLM2-135M**       | **49,152** | **~75M**      | **~9%**           |
| mistralai/Mistral-7B   | 32,000  | ~49M             | ~6%               |

SmolLM2 oferă cel mai bun echilibru: vocab suficient de mare pentru acoperire
bună (inclusiv cod și matematică), fără a "fura" capacitate de la layerele
transformer.

### 2. Antrenat pe exact datele noastre

SmolLM2 a fost antrenat pe `fineweb-edu-dedup` și `cosmopedia-v2` — exact
mixul nostru de pretraining din Faza 1. Asta înseamnă:

- compresia tokenică mai bună pe corpusul nostru față de tokenizerele antrenate
  pe web general,
- mai mulți tokeni utili per bloc de 2048,
- mai puțin padding efectiv pe secvențele scurte.

### 3. Byte-level BPE — zero probleme OOV

Orice secvență de bytes este reprezentabilă. Nu există token `[UNK]` practic.
Util în special când vom intra în Faza 2 cu date de reasoning și cod.

### 4. Compatibilitate HF fără licențe speciale

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
# vocab_size: 49152
# eos_token_id: 0
# bos_token_id: 1
```

Nu necesită aprobare de acces sau cont HF special, spre deosebire de Llama-3.

## Ce s-a actualizat în proiect

- `configs/dllm_1b_blueprint.json`:
  - `vocab_size: 64000` → `49152`
  - `tokenizer_name: "Qwen/Qwen2.5-0.5B"` → `"HuggingFaceTB/SmolLM2-135M"`

## Dacă vrem un tokenizer custom în viitor

Opțional, după ce avem date suficiente, putem antrena un tokenizer BPE
custom cu `tokenizers` (HF) pe corpusul nostru real. Procedura:

```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["data/corpus_sample.txt"],
    vocab_size=49152,
    min_frequency=2,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
)
tokenizer.save_model("tokenizer/")
```

Aceasta ar da o compresie și mai bună pe datele noastre specifice, dar
necesită un corpus curat pre-asamblat de minim câteva GB.

## Verificare rapidă

```bash
python -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M')
print(f'vocab_size: {len(t)}')
sample = 'The transformer architecture uses self-attention mechanisms.'
ids = t.encode(sample)
print(f'tokens: {len(ids)} pentru {len(sample.split())} cuvinte')
print(f'compression ratio: {len(sample.split()) / len(ids):.2f} words/token')
"
```

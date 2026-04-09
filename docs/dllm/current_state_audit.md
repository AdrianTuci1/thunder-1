# Audit Curent: dLLM From-Scratch (0-1B)

Data auditului: 2026-04-09

## Verdict rapid

Targetul real al proiectului trebuie inghetat asa:

- model diffusion language de aproximativ `0.8B-1B`,
- context `2048`,
- inferenta pe `RTX 4090`,
- `fast` in `5-10` iteratii,
- `thinking` cu mai multe iteratii,
- training in principal pe `A100 80GB`, folosind burst-uri scurte din credite gratuite.

Thunder nu este inca pregatit pentru acest obiectiv. Repo-ul actual este o fundatie utila pentru experimente pe un backbone existent, dar nu este inca un stack complet pentru:

- model bidirectional antrenat de la zero,
- pipeline de date verificat si versionat,
- training distribuit pe clustere,
- monitorizare si resume solide pentru rulaje lungi.

## Ce exista deja

- Adaptare la forward continuu pentru difuzie in [core/diffusion_model.py](../../core/diffusion_model.py).
- Scheduler de zgomot si obiective dedicate in [training/noise_scheduler.py](../../training/noise_scheduler.py) si [training/loss_functions.py](../../training/loss_functions.py).
- Motor de inferenta cu dynamic canvas pentru generare in [core/diffusion_engine.py](../../core/diffusion_engine.py).
- Baza pentru dynamic batching in [core/dynamic_batching.py](../../core/dynamic_batching.py).
- Mix de dataset-uri si normalizare de scheme in [training/data_pipeline.py](../../training/data_pipeline.py).

## Ce taiem acum ca sa nu ne risipim

Pentru acest proiect, urmatoarele directii sunt redundante sau premature:

- orice dependenta de `Qwen/Qwen3.5-9B` sau alte backbones mari pre-antrenate,
- LoRA ca strategie principala,
- tinta de context `8k`, `32k` sau mai mult,
- optimizari orientate strict pe serving inainte sa stabilizam trainingul,
- dataset mixing prea exotic inainte sa obtinem cursivitate pe corpusuri curate.

## Ce lipseste sau este doar partial

| Arie | Status | Observatie |
| --- | --- | --- |
| Training from scratch 0.8B-1B | Lipseste | Configul curent porneste din `Qwen/Qwen3.5-9B` si adauga LoRA; asta este adaptare pe model pre-antrenat, nu training de la zero. |
| Bidirectional / fara masca cauzala | Partial | Exista `is_causal=False`, `use_cache=False` si masca plina de `1`, dar nu exista teste de atentie care sa demonstreze ca fiecare strat ruleaza complet bidirectional. |
| Dynamic canvas | Partial | Exista la inferenta prin `max_new_tokens`; pentru targetul de `2048` este util, dar nu trebuie sa complice trainerul inainte sa avem packing si curriculum stabile. |
| Procesare text in blocuri paralele | Partial | Configul spune `packing=True`, dar pipeline-ul actual tokenizeaza exemplu-cu-exemplu si nu face packer real pe blocuri lungi, bucketizare de lungimi sau sharding paralel pentru trainer. |
| Compresie | Lipseste | Nu exista compresie pentru shard-uri de date, compresie latenta, activation compression sau optimizer-state sharding. |
| Integritate dataset-uri | Lipseste | `data/` nu contine corpusuri reale, nu exista manifest local cu checksum-uri, licente, schema si deduplicare. |
| Training distribuit | Lipseste | Nu exista entrypoint pentru `torchrun`, FSDP/ZeRO, Ray Train, DeepSpeed sau configuratii multi-node. |
| Resume / checkpoint complet | Partial | Checkpoint-urile salveaza acum si stare de optimizer/scheduler, dar nu exista inca un launcher distribuit care sa refaca automat tot contextul de rulare. |
| Monitorizare | Partial | Exista acum `metrics.jsonl` si script de status, dar lipsesc dashboard-uri GPU/system metrics, evaluari automate periodice si alerte. |
| Teste de consistenta | Partial | Exista teste de sintaxa, dar mai sunt fisiere care refera module inexistente sau API-uri invechite. |

## Concluzii tehnice pe cerintele tale

### 1. Dynamic canvas

Avem doar un inceput pentru inferenta. Pentru targetul de `2048`, dynamic canvas trebuie pastrat simplu:

- curriculum de lungime: 256 -> 512 -> 1024 -> 2048,
- bucketizare de secvente si packing real pe blocuri,
- decizie dinamica pe numar de pasi de difuzie in functie de lungime si dificultate.

### 2. Procesarea textului in blocuri paralele

Momentan nu este suficienta. Pentru un dLLM de 0.8B-1B trebuie:

- shard-uri mari pe documente,
- pre-tokenizare offline,
- block packing cu delimitatori de document,
- prefetch asincron si worker-i dedicati,
- layout de batch pe ferestre similare ca lungime,
- eventual blockwise denoising pe micro-batches pentru ferestre lungi.

### 3. Fara masca cauzala, bidirectional

Directia este corecta, dar validarea lipseste. Avem nevoie de:

- test care verifica explicit matricea de atentie fara triunghi superior mascat,
- audit pe RoPE/pozitionare pentru efecte ramase din regimul autoregresiv,
- un backbone gandit de la inceput pentru bidirectional diffusion, nu doar "decoder cu `is_causal=False`".

### 4. Compresie si alte imbunatatiri noi

Imbunatatiri recomandate:

- compresie de date la nivel de shard cu `zstd` sau `gzip` si checksum per shard,
- latent bridge cu compresie 2:1 sau 4:1 inainte de difuzie, apoi projector invers,
- activation checkpointing agresiv,
- optimizer/state sharding pe multi-node,
- cache de tokenizare si deduplicare pe documente,
- curriculum pe timesteps si pe lungime,
- evaluare pe bucket-uri de zgomot pentru a vedea unde se rupe reconstructia,
- distilare de sampling pentru `fast=5-10` iteratii si `thinking=16-32`.

## Probleme concrete observate in repo

- Trainerul era configurat inconsistent si nu salva stare completa de resume; a fost corectat in [training/diffusion_lm_trainer.py](../../training/diffusion_lm_trainer.py).
- `core/dynamic_batching.py` avea importuri lipsa; au fost adaugate.
- `TimestepEmbedder` folosea cheia gresita pentru numarul de pasi de difuzie; a fost corectata in [core/diffusion_model.py](../../core/diffusion_model.py).
- Exista referinte vechi catre module care nu mai exista sau nu sunt in repo, mai ales in unele teste si fisiere de documentare.

## Ce trebuie sa existe inainte de primul training serios

1. O configuratie dedicata pentru modelul 0.8B-1B, context 2048, separata de Qwen/LoRA.
2. Manifest de dataset local cu checksum si licenta pentru fiecare shard.
3. Launcher distribuit (`torchrun` sau Ray/Anyscale job) cu resume real.
4. Checkpointing complet: model, optimizer, scheduler, RNG, sampler state.
5. Metrics structurate si un script de status.
6. Evaluari periodice pe reconstructie, coerenta si throughput.

## Fisiere noi pentru audit si operare

- [configs/dllm_1b_blueprint.json](../../configs/dllm_1b_blueprint.json)
- [data/manifests/dllm_corpus_manifest.example.json](../../data/manifests/dllm_corpus_manifest.example.json)
- [scripts/audit_training_readiness.py](../../scripts/audit_training_readiness.py)
- [scripts/verify_dataset_integrity.py](../../scripts/verify_dataset_integrity.py)
- [scripts/report_training_status.py](../../scripts/report_training_status.py)
- [scripts/preflight_dllm.sh](../../scripts/preflight_dllm.sh)

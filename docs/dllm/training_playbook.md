# Playbook: dLLM 0.85B, Context 2048, Fast 5-10 Steps

Data: 2026-04-09

## Scope fix

De aici inainte mergem pe un singur obiectiv:

- aproximativ `0.85B` parametri,
- context `2048`,
- `fast` in `5-10` iteratii,
- `thinking` in `16-32` iteratii,
- inferenta pe `RTX 4090`,
- training pe `A100 80GB`,
- buget efectiv format din credite gratuite cumulate.

Tot ce nu ajuta direct acest scope trebuie amanat.

## Pas cu pas

### Pasul 1. Inghata starea curenta si ruleaza preflight

Ruleaza:

```bash
./scripts/preflight_dllm.sh
```

Scop:

- sa vezi ce avem deja,
- sa confirmi lipsurile de training from scratch,
- sa verifici manifestul de dataset.

### Pasul 2. Defineste blueprint-ul pentru modelul 0.85B

Porneste de la [configs/dllm_1b_blueprint.json](../../configs/dllm_1b_blueprint.json).

Recomandare pentru primul target:

- `~850M-950M` parametri,
- `hidden_size` ~2048,
- `24-28` straturi,
- `16` heads,
- context final `2048`,
- masca non-cauzala din prima iteratie,
- `x0` prediction + self-conditioning,
- distilare ulterioara pentru sampling rapid.

### Pasul 3. Pune dataset-urile sub control strict

1. Copiaza [data/manifests/dllm_corpus_manifest.example.json](../../data/manifests/dllm_corpus_manifest.example.json) in manifestul tau local.
2. Completeaza pentru fiecare shard:
   - path,
   - format,
   - checksum,
   - licenta,
   - cheile obligatorii.
3. Ruleaza:

```bash
python3 scripts/verify_dataset_integrity.py --manifest data/manifests/dllm_corpus_manifest.local.json --strict
```

Ce trebuie sa urmaresti:

- checksum corect,
- zero fisiere lipsa,
- zero shard-uri goale,
- schema consistenta,
- fara duplicate evidente intre corpusuri.

### Pasul 4. Construieste pipeline-ul de date pentru blocuri paralele

Obligatoriu inainte de un run mare:

- tokenizare offline pe shard-uri,
- packing real pe blocuri fixe,
- delimitatori de document,
- bucketizare pe lungime,
- `num_workers` separat de batch size,
- cache local al tokenilor,
- compresie a shard-urilor.

### Pasul 5. Fa curriculum-ul de training

Ordinea recomandata:

1. Short-context warmup: 256 tokeni.
2. Mid-context: 512 tokeni.
3. Consolidare: 1024 tokeni.
4. Target final: 2048 tokeni.
5. Mix principal: text curat + instruct + reasoning.

Nu sari direct la 2048 cu modelul complet daca nu ai:

- loss stabil,
- reconstructie buna pe secvente scurte,
- checkpoint resume verificat,
- throughput constant.

### Pasul 6. Alege hardware-ul

Estimare inginereasca, nu masuratoare directa:

| Target | Profil minim util | Profil recomandat |
| --- | --- | --- |
| Pilot 150M-300M | 1x A100 80GB | 1x A100 80GB |
| Pre-productie 400M-600M | 2x A100 80GB | 4x A100 80GB |
| Tinta 0.85B-1B | 4x A100 80GB | 8x A100 80GB |

De ce A100 80GB:

- memorie suficienta pentru BF16 + activari lungi,
- disponibilitate buna in platforme cloud,
- maturitate pentru FSDP/NCCL,
- echilibru bun intre cost si stabilitate.

H100 este mai rapid daca bugetul permite, dar A100 ramane alegerea pragmatica pentru un proiect bazat pe credite gratuite.

### Pasul 7. Nu sari direct la modelul final

Strategia care reduce riscul de "text aleator" este:

1. pilot mic: 150M-300M,
2. validare de cursivitate si reconstructie,
3. crestere la 400M-600M,
4. abia apoi run-ul 0.85B-1B.

Astfel verifici devreme daca arhitectura si loss-ul chiar invata limba, nu doar memoreaza zgomot.

### Pasul 8. Porneste cu un run de validare, nu cu run-ul final

Primul run trebuie sa fie scurt:

- 200-1000 pasi,
- subset mic de corpus,
- checkpoint la fiecare 100-200 pasi,
- validare reconstructie,
- metrics pe fiecare pas de optimizer.

Abia dupa asta urci modelul, contextul, batch-ul global si corpusul complet.

## Cum ne asiguram ca nu iese text aleator

Acesta este criteriul central al proiectului. Ordinea corecta este:

1. Corpus curat si divers, nu zgomot brut.
2. Reconstructie buna la timesteps mici si medii.
3. Sample decode periodic pe set de validare fix.
4. Distilare a samplingului abia dupa ce teacher-ul merge.

Semnale bune:

- reconstructie stabila pe mostre fixe,
- scadere clara a `denoising_loss`,
- mostre care devin gramaticale inainte sa devina "smart",
- consistenta intre run-uri reluate din checkpoint.

Semnale rele:

- repetitii si loops,
- cuvinte valide dar fara sintaxa,
- scadere de loss fara imbunatatire in sample-uri,
- sensibilitate mare la seed dupa putini pasi.

## Ce monitorizam obligatoriu

### Metrics de model

- `loss`
- `denoising_loss`
- reconstructie top-1 / top-k din `x0`
- loss pe bucket-uri de timestep
- norme de activari si grad norm
- rata de token collapse / repetitie
- rata de reconstructie pentru setul fix de probe

### Metrics de sistem

- tokens/sec
- samples/sec
- step time
- GPU memory
- GPU utilization
- CPU dataloader wait
- disk throughput
- lag intre checkpoint-uri

### Artefacte minime per run

- `metrics.jsonl`
- log brut
- checkpoint-uri versionate
- config exact al run-ului
- sample-uri de reconstructie la interval fix

Scriptul local de status:

```bash
python3 scripts/report_training_status.py --run-dir thunder_qwen_32k
```

## Ce modificam daca oprim training-ul

### Caz 1. NaN / inf in loss

Modifica imediat:

- scade `learning_rate` de 2x-4x,
- foloseste BF16 pe A100, nu FP16,
- creste frecventa checkpoint-urilor,
- dezactiveaza temporar self-conditioning daca explodeaza devreme,
- verifica normele pe embeddings si `x0_head`.

### Caz 2. OOM sau throughput foarte slab

Modifica:

- context mai scurt,
- batch per device mai mic,
- gradient accumulation mai mare,
- packing mai agresiv,
- checkpointing de activari,
- compresie de shard-uri si pre-tokenizare offline,
- mai multe worker groups in cluster,
- latent bridge mai comprimat.

### Caz 3. Loss scade, dar textul ramane prost

Modifica:

- raportul dintre obiectivele de reconstructie si semantic anchor,
- diversitatea dataset-ului,
- temperatura / rounding / clamp,
- curriculum-ul pe zgomot,
- evaluarea pe bucket-uri de timestep ca sa vezi unde se rupe,
- nu distila la 5-10 pasi pana nu ai teacher bun la mai multi pasi.

### Caz 4. Overfitting pe subset mic

Modifica:

- deduplicarea,
- mixul de dataset-uri,
- regularizarea,
- frecventa evaluarilor,
- opreste run-ul si revino cu corpus mai mare sau mai curat.

### Caz 5. Reconstructie buna la short-context, proasta la 2048

Modifica:

- creste contextul gradual,
- foloseste parallel block packing,
- adauga blocuri cu overlap si boundary tokens,
- verifica daca dynamic canvas este activ si la training, nu doar la inferenta.

## Ordinea recomandata de executie in proiect

1. Preflight si audit.
2. Manifest si integritate dataset.
3. Pilot 150M-300M.
4. Blueprint model 0.85B.
5. Launcher distribuit: [scripts/launch_train_torchrun.sh](../../scripts/launch_train_torchrun.sh).
6. Metrics + checkpointing + resume.
7. Micro-run pe A100.
8. Run final 0.85B-1B.

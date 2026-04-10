# Hugging Face Dataset Plan

Scopul nu este sa aruncam cat mai multe dataset-uri in mixer, ci sa obtinem:

1. cursivitate,
2. reconstructie stabila,
3. reasoning decent fara colaps,
4. cost control pentru A100-uri obtinute din credite gratuite.

## Faza 1: Pretraining de baza

Mixul recomandat pentru primele rulaje:

- `HuggingFaceTB/smollm-corpus` subset `fineweb-edu-dedup`
  Motiv: web educational deduplicat, foarte bun pentru fluenta si cunostinte generale curate.
- `HuggingFaceTB/smollm-corpus` subset `cosmopedia-v2`
  Motiv: text educational si explicativ sintetic, util pentru formulare clara si coerenta.
- `open-web-math/open-web-math`
  Motiv: adauga densitate pentru probleme matematice si rationamente simbolice.
- `codeparrot/codeparrot-clean`
  Motiv: cod curat pentru a nu rata complet generalizarea pe snippets si instructiuni tehnice.

## Faza 2: Post-training / alignment dupa baza

Abia dupa ce modelul produce text fluent:

- `Open-Orca/SlimOrca`
- `HuggingFaceH4/ultrafeedback_binarized` split `train_sft`
- `open-thoughts/OpenThoughts-114k`

Ideea este sa nu incepem direct cu prea mult synthetic reasoning inainte sa avem o baza lingvistica solida.

## Raport initial recomandat

Pentru pretraining:

- `55%` FineWeb-Edu-Dedup
- `20%` Cosmopedia v2
- `15%` OpenWebMath
- `10%` CodeParrot Clean

Pentru post-training:

- `35%` SlimOrca
- `35%` UltraFeedback SFT
- `20%` OpenThoughts
- `10%` CodeParrot Clean

## Ce evitam la inceput

- corpusuri uriase dar foarte zgomotoase,
- prea mult code daca scopul principal este cursivitatea generala,
- prea mult chain-of-thought sintetic inainte de stabilizarea limbajului,
- dataset-uri gated sau greu de operat daca exista variante mai simple pentru primele rulaje.

## Regula practica

Daca bugetul real ramane in plaja `600-1000 USD` in credite cumulate, este mai eficient sa:

1. validam arhitectura pe subseturi curate,
2. urcam treptat numarul de tokeni,
3. pastram reasoning-ul greu pentru faza a doua,
4. distilam sampling-ul rapid doar dupa ce teacher-ul merge bine.

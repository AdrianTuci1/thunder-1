# Ghid Platforme: Replicate, Anyscale, Saturn Cloud

Actualizat la: 2026-04-09

Acest ghid rezuma modul practic de pornire pentru antrenament sau staging de training pe baza documentatiei oficiale curente. Pentru pasi UI/API specifici, urmeaza linkurile oficiale de la final.

## Recomandare rapida

- Pentru training distribuit serios: Anyscale.
- Pentru prototipare interactiva si apoi job-uri: Saturn Cloud.
- Pentru deploy/servire a unui model impachetat sau staging simplu: Replicate.
- Pentru inferenta finala locala: RTX 4090.

## Pregatire comuna, indiferent de platforma

Inainte sa urci proiectul:

1. Ruleaza `./scripts/preflight_dllm.sh`.
2. Verifica manifestul de date.
3. Fa un micro-run local sau pe o singura masina.
4. Stabileste clar unde scrii checkpoint-urile si logurile.
5. Pastreaza configul exact al run-ului in acelasi folder cu checkpoint-urile.

## Strategie pentru credite gratuite

Nu trata creditele ca pe un singur run gigantic. Foloseste-le in trei faze:

1. Faza A: smoke tests si pilot mic pe orice platforma iti da acces rapid la GPU.
2. Faza B: run de validare pe `1x A100 80GB` pentru a confirma cursivitatea.
3. Faza C: burst scurt pe `4x` sau `8x A100 80GB` pentru run-ul final.

Regula practica:

- nu consuma creditele scumpe pana nu ai sample-uri bune pe pilot,
- muta run-ul mare numai dupa ce resume-ul merge,
- pastreaza artefactele portabile intre platforme: manifest, config, checkpoint, metrics.

## Replicate

Replicate este foarte bun pentru ambalarea unui model custom si rularea lui ca deployment cu autoscaling. Pentru training multi-node de la zero pana la 1B, este in general mai natural sa folosesti Anyscale sau Saturn. Foloseste Replicate mai ales pentru:

- staging de training pe o singura instanta,
- validare de packaging,
- deploy al modelului dupa antrenare.

### Flux recomandat

1. Impachetezi modelul si codul de rulare in formatul compatibil Replicate/Cog.
2. Creezi modelul tau pe platforma.
3. Creezi un deployment din model sau dintr-o versiune a modelului.
4. In `Settings` alegi hardware-ul si limitele de autoscaling.
5. Rulezi predictii sau job-uri scurte si verifici monitorizarea deployment-ului.

### Cand il alegi

- cand vrei cea mai rapida cale spre un deployment GPU,
- cand trainingul complet nu are nevoie de multi-node complex,
- cand te intereseaza mai mult serving-ul decat orchestration-ul de cluster.

## Anyscale

Anyscale este alegerea naturala pentru training distribuit pe Ray clusters si job-uri batch. Pentru un dLLM de 1B, aici ai cel mai clar model de operare:

### Flux recomandat

1. Creezi un workspace cu codul proiectului.
2. Definesti un `compute config` cu head node si worker groups GPU.
3. Alegi instanta GPU pentru worker group-urile de training.
4. Definesti imaginea/containerul si entrypoint-ul jobului.
5. Submisi job-ul catre cluster.
6. Monitorizezi din pagina jobului: status, logs, metrics, notifications si Ray Dashboard.

### Cand il alegi

- cand vrei multi-node real,
- cand vrei sa scalezi de la validare la run mare fara sa schimbi paradigma,
- cand vrei monitorizare si operare mai apropiate de un cluster ML clasic.

## Saturn Cloud

Saturn Cloud este foarte bun cand vrei sa pornesti cu un notebook/Jupyter GPU, sa validezi rapid totul si apoi sa clonezi configuratia intr-un job de training.

### Flux recomandat

1. Creezi un resource de tip Jupyter Server sau Job.
2. Alegi imaginea si resursele GPU.
3. Faci upload la cod sau conectezi repo-ul.
4. Rulezi un smoke test pe o singura masina.
5. Clonezi intr-un Job sau folosesti ghidul lor pentru training multi-node cu `torchrun`.
6. Monitorizezi statusul resursei, logurile si artefactele scrise de job.

### Cand il alegi

- cand vrei iteratie rapida cu notebook + job,
- cand vrei infrastructura GPU preconfigurata fara prea multa frictiune,
- cand pipeline-ul tau poate evolua de la explorare la productie incremental.

## Ce cluster as porni pentru primul run serios

Pentru targetul nostru, aproximativ `0.85B-1B` si context `2048`:

- preferat: 8x A100 80GB,
- acceptabil pentru run-uri mai conservatoare: 4x A100 80GB,
- pentru smoke test: 1x A100 80GB.

Configuratie de start recomandata:

- BF16,
- activation checkpointing activ,
- checkpoint la 200-500 pasi la inceput,
- corpus redus pentru primele rulaje,
- loguri si metrics separate de datele brute.

## Monitorizare pe platforma

Indiferent de platforma, urmaresti:

- utilizarea GPU,
- memorie GPU,
- logs ale jobului,
- rata de erori / retry,
- latenta per step sau per batch,
- checkpoint-uri noi aparute la timp.

Pe Anyscale monitorizezi in primul rand job page + metrics + logs + Ray Dashboard.

Pe Replicate monitorizezi deployment-ul si setarile de hardware/autoscaling.

Pe Saturn Cloud monitorizezi resource/job logs si artefactele scrise de codul tau.

## Surse oficiale

- Replicate: [Create a deployment](https://replicate.com/docs/topics/deployments/create-a-deployment)
- Replicate: [Model hardware](https://replicate.com/docs/topics/models/hardware)
- Anyscale: [Compute configuration](https://docs.anyscale.com/configuration/compute)
- Anyscale: [Get started with jobs](https://docs.anyscale.com/tutorials/submit-a-job)
- Anyscale: [Monitor a job](https://docs.anyscale.com/jobs/monitor)
- Saturn Cloud: [Resources overview](https://saturncloud.io/docs/user-guide/how-to/resources/)
- Saturn Cloud: [Create a Dask cluster](https://saturncloud.io/docs/user-guide/how-to/scale/create_dask_cluster/)
- Saturn Cloud: [Multi-node multi-GPU parallel training](https://saturncloud.io/docs/llms/parallel_training/)
- Saturn Cloud: [GPU fractionalization / supported GPU models](https://saturncloud.io/docs/gpu-management/gpu-fractionalization/)

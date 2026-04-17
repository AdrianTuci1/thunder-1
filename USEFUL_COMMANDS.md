# ⚡ Thunder Training: Comenzi Utile

Acest document reunește comenzile esențiale pentru gestionarea stației de lucru A100 și a procesului de antrenament.

## 📺 Gestionare Sesiuni (Screen)

Sesiunile `screen` îți permit să rulezi codul în fundal, chiar dacă te deconectezi.

- **Pornire sesiune nouă**:
  ```bash
  screen -S thunder_train
  ```
- **Detașare (Background)**:
  Apasă `Ctrl + A`, apoi tasta `D`.
- **Pornire direct în fundal (Detached)**:
  ```bash
  screen -d -m -S thunder_train accelerate launch training/diffusion_lm_trainer.py
  ```
- **Reatașare (Reia vizualizarea)**:
  ```bash
  screen -r thunder_train
  ```
- **Listare sesiuni active**:
  ```bash
  screen -ls
  ```
- **Închidere forțată sesiune (Kill)**:
  ```bash
  screen -X -S thunder_train quit
  ```
- **Curățare sesiuni "moarte"**:
  ```bash
  screen -wipe
  ```

## 🚀 Antrenament și Monitorizare

- **Lansare Antrenament (Accelerate)**:
  ```bash
  accelerate launch training/diffusion_lm_trainer.py
  ```
- **Monitorizare GPU (Live)**:
  ```bash
  nvtop
  ```
  *Dacă nu este instalat:* `watch -n 1 nvidia-smi`
- **Verificare resurse RAM/CPU**:
  ```bash
  htop
  ```

## 🛠️ Depanare și Mediu

- **Verificare Mediu (.env)**:
  ```bash
  python3 -c "from dotenv import load_dotenv; load_dotenv(); import os; print(f'WandB: {bool(os.getenv(\"WANDB_API_KEY\"))}, HF: {bool(os.getenv(\"HF_TOKEN\"))}')"
  ```
- **Resetare CUDA Cache**:
  Dacă primești erori neașteptate de memorie, încearcă să închizi procesul python:
  ```bash
  pkill -9 python3
  ```

## 📦 Transfer Proiect

- **Împachetare cod**:
  ```bash
  bash scripts/package_thunder.sh
  ```

python3 scripts/preview_checkpoint.py
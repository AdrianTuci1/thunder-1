# Ghid Simplificat Thunder dLLM ⚡

Acest ghid presupune că ai folosit **Startup Script-ul** și **SSH Key** la configurarea instanței pe Vultr.

## Pasul 1: Conectarea și Pregătirea
Loghează-te pe server (SSH):
```bash
ssh root@<IP_SERVER>
```
*Asigură-te că ai completat **WANDB_API_KEY** în fișierul `.env` pentru monitorizare automată.*

## Pasul 2: Transferul Codului (de pe Mac-ul tău local)
Deschide un terminal nou pe Mac-ul tău (unde ai fișierele) și rulează:
```bash
scp thunder_deploy.tar.gz root@<IP_SERVER>:/root/
scp .env root@<IP_SERVER>:/root/
```

## Pasul 3: Instalarea Mediului Python
Pe server (reîntoarce-te în fereastra de SSH):
```bash
tar -xzf thunder_deploy.tar.gz
cd thunder
bash setup_env.sh
```

## Pasul 4: Pornirea Antrenamentului (mod PRO)
Vom folosi o sesiune `screen` pentru stabilitate și `accelerate` pentru viteză:
```bash
screen -S thunder_run
# Înăuntru tastează:
python3 training/diffusion_lm_trainer.py
```
*   **Monitorizare Real-Time:** `nvtop` (într-o altă fereastră SSH) pentru GPU.
*   **Monitorizare Cloud:** Verifică dashboard-ul tău pe [wandb.ai](https://wandb.ai).
*   **Ieșire din ecran:** `Ctrl+A` apoi `D`.
*   **Revenire:** `screen -r thunder_run`.

---
⚡ **Spor la procesat miliarde de tokeni!**

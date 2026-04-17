# 🧠 Ghid Generare Date Sintetice (SFT)

Acest document explică procesul de generare a celor 50.000 de perechi de date instrucțiune-răspuns pentru fine-tuning-ul modelului Thunder.

## 🛠️ Pregătire Mediu

Înainte de prima rulare, instalează bibliotecile necesare:
```bash
pip install google-generativeai tqdm
```

## 🚀 Pași de Rulare

### Varianta A: Pe mașina locală (Mac)
Ideală pentru monitorizare rapidă.
1. `export GEMINI_API_KEY="cheia_ta_aici"`
2. `python scripts/sft/generate_synthetic_islands.py`

### Varianta B: Pe mașina virtuală (VM / Screen)
Ideală pentru generări de lungă durată fără a depinde de laptop.
1. `screen -S sft_gen`
2. `export GEMINI_API_KEY="cheia_ta_aici"`
3. `python scripts/sft/generate_synthetic_islands.py`
4. Detașare: `Ctrl + A`, apoi `D`.

## 📦 Rezultat și Stocare
*   **Fișier**: `data/synthetic_sft_english_50k.jsonl`
*   **Mărime estimată**: **~70MB - 100MB** (55k perechi).
*   **Checkpointing**: Scriptul salvează fiecare linie imediat. Dacă se întrerupe, relansează-l și va continua de unde a rămas.

## 💰 Estimare Costuri (Gemini 3.1 Flash)

Calculul este bazat pe generarea a **50.000 de perechi**.

| Componentă | Estimare (Tokens) | Preț (per 1M) | Cost Total |
| :--- | :--- | :--- | :--- |
| **Input (Prompts)** | ~2.5M tokens | $0.10 | ~$0.25 |
| **Output (Răspunsuri)** | ~10M - 15M tokens | $0.40 | ~$4.00 - $6.00 |
| **TOTAL ESTIMAT** | **~15M tokens** | - | **~$4.25 - $6.25** |

> [!NOTE]
> **Gemini 3.1 Flash** (Modelul recomandat în 2026) oferă o calitate mult superioară în generarea de cod Python/SQL față de generațiile anterioare, menținând costurile extrem de accesibile pentru volume mari de date.

> [!TIP]
> Bugetul de **7$** este mai mult decât suficient pentru întregul dataset de 50k perechi folosind cel mai nou model Flash.

---

## 🛠️ Strategia "Islands"
Fiecare "insulă" de date acoperă un domeniu specific:
- **Science**: Fizică, Matematică, Biologie.
- **Coding**: Python, SQL, algoritmi.
- **Identity**: Alinierea modelului cu brand-ul **StaticLabs** și arhitectura **Thunder**.
- **Logic**: Probleme de gândire critică.

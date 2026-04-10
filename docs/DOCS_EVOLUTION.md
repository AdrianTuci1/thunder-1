# ⚡ Thunder 1: Roadmap Strategic (Bilingv & Aliniat)

Acest document descrie evoluția Thunder 1 de la un pilot experimental la un asistent bilingv (Română/Engleză) optimizat pentru interfețe moderne precum OpenWebUI.

## 1. Strategia Multilingvă (RO + EN)
Deși modelul are 0.8B parametri, el poate gestiona impecabil ambele limbi folosind un transfer de cunoștințe (Cross-lingual transfer).

- **Data Mix**: 70% FineWeb-Edu (English) | 25% FuLG/CulturaX (Romanian) | 5% Identity/Specialized (Staticlabs).
- **Tokenizer**: Rămânem pe SmolLM2 (49k vocab) - este foarte eficient pentru ambele limbi.

## 2. Pipeline-ul de Date pentru "The Big Run"
Pentru rularea de 24h-48h, vom folosi următoarele surse din Hugging Face prin sistemul nostru de `StreamingDataset`:

| Sursă | Limbă | Tip | Rol |
| :--- | :--- | :--- | :--- |
| `HuggingFaceFW/fineweb-edu` | EN | Web | Logică, Raționament, Știință |
| `readerbench/FuLG` | RO | Web/Crawl | Vocabular curent, Context local |
| `wikipedia (ro)` | RO | Enciclopedic | Fapte corecte, Istorie, Cultură |
| `Identity (Custom)` | RO/EN | Sintetic | Personalitate Staticlabs |

## 3. Etapa de "Maturizare": SFT & DPO
Odată ce pre-trainingul este gata, Thunder va trece prin:

### SFT (Supervised Fine-Tuning)
- **Dataset**: Sintetizat cu Claude 3.5 / GPT-4o.
- **Focus**: Formatare Markdown (Tabele, Cod, Bold).
- **Alignment**: Învățarea sintaxei OpenWebUI pentru "file chips" și componente vizuale.

### DPO (Direct Preference Optimization)
- **Scop**: Eliminarea halucinațiilor și a comportamentelor nedorite.
- **Tone**: Professional, helpful, Staticlabs-oriented.

## 4. Arhitectură & Infrastructură
- **GPU**: Modal (L40S / A100).
- **Optimizări**: GQA (Grouped Query Attention) pentru inferență rapidă și context de 8k.
- **Deploy**: DigitalOcean (pentru API-ul final) sau auto-hosted în OpenWebUI.

---
*Creat de Antigravity pentru Thunder 1 Project (Staticlabs)*

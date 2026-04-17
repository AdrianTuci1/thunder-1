# 📂 Thunder Scripts Directory

Această structură organizează utilitarele proiectului pe categorii funcționale pentru a facilita dezvoltarea și scalarea.

## 🏗️ Structură Directoare

### [prelaunch/](./prelaunch)
Scripturi de audit și validare ce trebuie rulate **înainte** de a porni un antrenament costisitor.
- `audit_training_readiness.py`: Verifică dacă toate componentele (model, date, config) sunt aliniate.
- `verify_hf_dataset_sources.py`: Testează conexiunea și formatul seturilor de date din HuggingFace.

### [pretraining/](./pretraining)
Scripturi pentru execuția și monitorizarea fazei de Foundation Training.
- `launch_train_torchrun.sh`: Scriptul principal pentru antrenament distribuit pe GPU-uri locale (A100/H100).
- `modal_train.py`: Integrarea pentru antrenament în cloud folosind Modal.
- `report_training_status.py`: Generarea de rapoarte umane bazate pe `metrics.jsonl`.

### [sft/](./sft)
Supervised Fine-Tuning și pregătirea modelului ca asistent.
- `generate_synthetic_islands.py`: Generator de date instrucțiune-răspuns folosind Gemini.
- `verify_dataset_integrity.py`: Verifică calitatea și formatul datelor generate.

### [inference/](./inference)
Testarea modelului și generarea de text.
- `local_inference.py`: **[RECOMANDAT]** Inferență rapidă pe VM/Local cu suport R2.
- `modal_inference.py`: Testare în cloud pe GPU L40S.
- `preview_checkpoint.py`: Inspectare rapidă a greutăților și configurației.

### [utils/](./utils)
Utilitare diverse.
- `cleanup_r2.py`: Gestionarea spațiului în Cloudflare R2.
- `test_r2_connection.py`: Verificarea rapidă a credențialelor de stocare.
- `package_thunder.sh`: Crearea de arhive pentru deployment rapid pe Vultr/Alte VM-uri.

---
**Notă:** Toate scripturile Python din aceste directoare trebuie rulate din rădăcina proiectului (ex: `python scripts/inference/local_inference.py`).

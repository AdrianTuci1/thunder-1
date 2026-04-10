# Plan de Implementare: Diffusion-LM pe Llama-3.2-3B via PrefixLM

Acest plan detaliază pașii pentru arhitectura de text diffusion continuu (inspirat din Li et al., 2022), dar păstrăm ca "creier" modelul **Llama-3.2-3B** prin adaptarea lui la un regim **bidirecțional** folosind tehnica **PrefixLM**.

Aceasta este o soluție foarte elegantă! Llama-3.2 are o putere de înțelegere și "reasoning" net superioară modelelor vechi de tip RoBERTa/DeBERTa. Problema lui era că, arhitectural, fiecare token se putea uita doar în stânga (mască de atenție triunghiulară - cauzală), ceea ce distrugea procesul de difuzie globală.

Prin **PrefixLM**, modificăm masca de atenție a modelului astfel încât pe segmentul de text care ne interesează (acolo unde se face difuzia și denoise-ul), atenția să devină o matrice plină de `1` (all-to-all / complet bidirecțională), permițând modelului să "vadă" tot vectorul $x_t$ zgomotos simultan.

---

## Proposed Changes

### 1. The Denoising Backbone: Llama-3.2-3B cu PrefixLM

#### [NEW] [core/diffusion_model.py](file:///workspace/thunder/core/diffusion_model.py)
Acesta va fi noul `model_adapter.py`. 
- **Modificarea Măștii de Atenție (PrefixLM)**: Vom suprascrie felul în care HuggingFace pasează `attention_mask`. 
  - Standard Llama: `mask[i, j] = 1 if i >= j else 0`
  - PrefixLM pentru noi: `mask[:, :] = 1` (Atenție Bidirecțională Completă pe toată secvența de interes).
  - Vom dezactiva `use_cache=False` (KV-cache nu are sens într-un setup bidirecțional unde toată secvența e updatată de la un pas de difuzie la altul).

- **Eliminarea RoPE (Rotary Position Embeddings) Causal Bias**: Din moment ce RoPE-ul lui Llama este gândit să încurajeze generarea secvențială, atenția bidirecțională e posibil să confunde modelul inițial. Totuși, RoPE injectează informație pozițională absolută și relativă, deci modelul va învăța destul de repede din *fine-tuning* sub MSE Loss cum să pună piesele la loc bidirecțional. Vom asigura suportul pentru RoPE, dar prin fine-tuning își va adapta bias-ul cauzal.

- **Intrări modificate**: Modelul va lua ca intrare $x_t$ (vectori continui de zgomot de dimensiune `[batch, seq_len, hidden_size=3072]`) în loc de `input_ids`. Funcția forward va invoca direct `model.model(inputs_embeds=x_t, attention_mask=custom_mask)`.

- **Ieșire ($x_0$-parametrization)**: La finalul Transformer-ului, adăugăm un strat `nn.Linear(3072, 3072)` sau folosim direct ieșirile din ultimul hidden state pe care le antrenăm să aproximeze **vectorul originar complet dez-zgomotot $x_0$**. 

### 2. The Embedding Bridge (Learnable sau Fixed)
Pentru ca Llama să înțeleagă semnalul continuu $x_t$, trebuie să folosim fix setul lui de word embeddings natively trained (`model.get_input_embeddings()`).
- O luăm ca atare ca matrice ancoră. Atunci când calculăm pierderea la pasul $t=1$, ne așteptăm ca $\hat{x}_0$ să fie extrem de aproape de unul dintre vectorii rând din această matrice.
- "Clamping Trick-ul" se va face prin găsirea celui mai apropiat vecin (Cosine Similarity / L2) în matricea nativă de vocabular a lui Llama.

### 3. The Normalization Bridge (Critical for Stability)
Deoarece embeddings-urile lui Llama-3.2 au o varianță foarte mică (`std=0.02`), difuzia directă în acest spațiu este instabilă (duce la colapsul "!!!").
- **Standardizare**: Vom mapa spațiul de embedding al lui Llama (std 0.02) într-un spațiu standard $N(0, 1)$ prin divizarea cu factorul de scalare ($\sigma \approx 0.02$).
- **Inference/Generation**: Toate operațiile de zgomot se fac în spațiul $N(0, 1)$. La final, înainte de clamping, readucem vectorii în spațiul original al lui Llama.
- **Logit Scaling**: Pentru calculul `L_round`, vom scala produsul punct (dot product) pentru a asigura o distribuție softmax non-uniformă, permițând modelului să "comită" la tokeni reali.

### 4. Training Loop Validation (MSE Loss pe $x_0$)

#### [MODIFY] [training/loss_functions.py](file:///workspace/thunder/training/loss_functions.py)
- **Masked MSE Loss**: Adăugăm suport pentru `attention_mask` astfel încât modelul să nu învețe padding-ul (care altfel ar domina gradientul).
- **Standardized MSE**: Calculăm pierderea în spațiul proiectat $N(0, 1)$ pentru a echilibra gradienții între LoRA (backbone) și noile capete (`x0_head`).

#### [NEW] [training/diffusion_lm_trainer.py](file:///workspace/thunder/training/diffusion_lm_trainer.py)
Aici construim bucla PyTorch de la 0, specializată:
1. Preluăm un text și obținem $x_0$.
2. **Standardizăm $x_0$** (Mean 0, Std 1).
3. Adăugăm zgomot conform *sqrt schedule*.
4. Trecem prin PrefixLM -> Obținem $\hat{x}_0$.
5. Măsurăm MSE Loss ($\hat{x}_0$, $x_0$) în spațiul standardizat.
6. Aplicăm gradienți cu **Gradient Clipping** setat la `1.0`.

---

## Verification Plan

### Automated Tests
1. **Mask Check**: Să măsor logurile din interiorul Llama Attention pentru a mă asigura că sub funcția noastră custom forward, matricea `attention_weight` nu conține un $-\infty$ fraudulos pe triughiul superior (adică vreau să văd atenție înapoi / înainte fără opreliști).
2. **Gradient Flow**: Să confirmăm că trecând direct `inputs_embeds` continui prin model (fără `input_ids`), PyTorch reușește să întoarcă gradienții `loss.backward()` cu succes către `Embedding Bridge`.
3. **Gradient Balance Check**: Verificăm dacă LoRA și `x0_head` au gradienți de magnitudini similare (nu 1:1000).
4. **Unwrapping Check**: Confirmăm că $\hat{x}_0$ transformat înapoi în spațiul Llama este capabil să recupereze textul original prin `argmax`.

### Manual Fine-Tuning Setup
- Pornim un **Micro-Run de 100 pași** cu LoRA/Freeze Parțial (Ex: dezghețăm doar modulele K, V și Q sau ultimele 8 straturi) pe texte scurte pentru a verifica convergența lui MSE (care ar trebui să aibă o drop curve accelerată comparativ cu Cross-Entropy pe vocabular `128k`).

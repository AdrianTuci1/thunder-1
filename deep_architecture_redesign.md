# Deep Architecture Redesign: Text Diffusion pe Llama-3.2-3B

Faptul că loss-ul nu converge și coerența lipsește indică o problemă structurală profundă. Abordarea de a lua un LLM din librării (HuggingFace/Unsloth), a opri masca cauzală (`is_causal=False`) și a spera că învață difuzie se lovește de **limitări fundamentale ale modului în care LLM-urile sunt construite la nivel de bază**.

## De ce a eșuat abordarea curentă? (Diagnosis)

1. **Textul este Discret, Zgomotul este Continuu**:
   - Difuzia funcționează perfect pe pixeli (valori continue). Adaugi zgomot Gaussian pe un pixel [0.5, 0.2, 0.1], are sens.
   - Textul este compus din indici discreți (Token ID 1045). Dacă adaugi zgomot direct peste embedding-urile pre-antrenate ale lui Phi-4, modelul este "orbit". Spațiul latent al unui LLM pre-antrenat **nu este izotrop** (nu are o formă regulată în care zgomotul gaussian să ierte).
2. **Loss-ul incorect**:
   - Probabil folosim un mix hibrid de Cross-Entropy (pentru a ghici tokenul final) și $\epsilon$-prediction. LLM-ul este setat să scoată logits, nu să returneze un vector de zgomot pur în spațiul continuu.
3. **Mecanismele de Atenție (RoPE)**:
   - RoPE (Rotary Position Embeddings) din Phi-4 este matematic optimizat pentru relații secvențiale (stânga -> dreapta). Chiar dacă scoți masca cauzală, encodarea pozițională tot "trage" modelul spre a fi autoregresiv.

---

## Soluția Radicală: Continuous Latent Diffusion cu LM Projection

Trebuie să depășim limitările librăriilor clasice și să scriem o punte (Bridge) între **Spațiul Discret** (tokeni) și **Spațiul de Difuzie** (zgomot continuu). Nu mai forțăm Phi-4 să *genereze* difuzie direct din LM head. 

Iată conceptul pentru a menține ideea creată (folosirea unui creier pre-antrenat), dar a o face să funcționeze:

### 1. The Embedding Bridge (Spațiul de Tranziție)
Nu facem difuzie direct pe embeddings de bază ale modelului.
* **Componenta**: Construim o rețea mică auto-encoder (sau o proiecție liniară + LayerNorm) $E(x)$ care preia token-embeddings-urile lui Phi-4 și le mapează într-un spațiu continuu *normalizat* și *izotrop* (unde direcțiile sunt egale).
* În acest "**Continuous Embedding Space**" realizăm tot procesul de difuzie (adăugăm noise).

### 2. Custom Forward Engine (Partial Freezing pe Llama-3.2-3B)
Llama are un design mult mai standard și stabil. Folosind acest model, adoptăm o strategie de **înghețare parțială**:
* **Înghețăm** complet primele straturi (ex: 0-16). Ele acționează ca un Feature Extractor universal perfect antrenat.
* **Antrenăm** doar "Bridge-ul" și ultimele straturi (16-32) fie full fine-tuning, fie printr-un LoRA cu rank foarte mare (r=256+).
* Ieșirea modelului este o predicție a latenților curați în *Spațiul de Tranziție*, **NU logits**.

### 2.1 Enforcing Isotropy (Normalization Layer)
Imediat după "Embedding Bridge" adăugăm un `LayerNorm` strict, fără parametri de învățare (sau cu weight/bias limitate restrictiv).
* Scopul: Să forțăm latenții proiectați într-o distribuție normală $N(0, I)$ înainte de a aplica zgomotul de difuzie. Spațiile de LLM sunt adesea "spiky"; LayerNorm-ul taie acele extreme (outliers) făcând spațiul prietenos pentru difuzia gaussiană.

### 3. The Rounding Step (Cristalizarea)
Cum transformăm latenții dez-zgomotoți înapoi în text la finalul inferenței?
* Latenții finali sunt trecuți printr-un proector invers către spațiul de embeddings al vocabularului.
* Folosim un algoritm de "Softmax/Rounding" care alege cel mai apropiat Token ID pe baza distanței Cosine sau L2.

### 4. Loss Function-ul Salvator & Semantic Anchor
Pierderea (Loss) nu mai trebuie să fie Cross-Entropy (ghicirea cuvântului perfect).
* **Diffusion MSE Loss**: Mean Squared Error direct în spațiul continuu al latenților (`predicted_noise` vs `true_noise`).
* **Semantic Anchor Loss**: Aceasta este cheia. O componentă a loss-ului de tip `L2` sau `Cosine Similarity` care penalizează latenții complet de-zgomotoți dacă deviază masiv de la un "Anchor". Ancora este setul de embeddings originale date de Llama pentru textul perfect. Astfel, forțăm bridge-ul de difuzie să respecte cunoștințele pre-antrenate (inteligența nativă a rețelei), prevenind modelul din a inventa un spațiu latent privat pe care doar el îl înțelege.

---

## Implementarea Practică (Independent de HuggingFace/Transformers)

Pentru a reuși acest lucru fără limitările librăriei `transformers`, trebuie să scriem arhitectura custom în PyTorch brut pe deasupra componentelor Llama-3.2:

1. Încărcăm doar greutățile de la `embedding_layer` și `transformer_blocks` ale lui Llama. Aruncăm complet `lm_head`.
2. Dezactivăm RoPE cauzal sau îl modificăm.
3. Transformăm [diffusion_engine.py](file:///Users/adriantucicovenco/Proiecte/thunder/core/diffusion_engine.py) astfel încât să antreneze folosind MSE + Semantic Anchor Loss.

Aceasta este direcția pe care au explorat-o studii ca **Diffusion-LM** și **Plaid**. Pleacă de la un model pre-antrenat (precum BERT sau RoBERTa - modele care *deja* sunt non-cauzale) sau adaptează forțat un model tip GPT printr-un strat gros de "continuous mapping".

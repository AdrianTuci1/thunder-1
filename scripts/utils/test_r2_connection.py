import os
import sys
import time
from dotenv import load_dotenv

# Adaugam root-ul proiectului in path pentru a putea incarca core/
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.config_manager import THUNDER_CONFIG
from core.storage import ObjectStorageManager

def run_test():
    # Incarcam variabilele din .env
    load_dotenv(".env")
    
    print("🧪 Testare conexiune Cloudflare R2...")
    
    # Fortam enabled pe True pentru test
    THUNDER_CONFIG["storage"]["enabled"] = True
    
    # Initializam managerul de stocare
    storage = ObjectStorageManager(THUNDER_CONFIG)
    
    if not storage.enabled:
        print("❌ Sincronizarea este dezactivata. Verifica daca ai .env si daca variabilele sunt corecte.")
        return

    # Cream un mic fisier de test
    test_dir = "scratch/test_r2_sync"
    os.makedirs(test_dir, exist_ok=True)
    file_path = os.path.join(test_dir, "hello_thunder.txt")
    
    with open(file_path, "w") as f:
        f.write("Salut! Daca vezi acest fisier, conexiunea R2 a modelului Thunder functioneaza perfect.\n")
        f.write(f"Test rulat la: {time.ctime()}")

    print(f"🚀 Incarcam fisierul de test in bucket-ul: {storage.bucket_name}...")
    storage.upload_checkpoint_async(test_dir)
    
    # Asteptam putin thread-ul de background sa termine
    print("⏳ Finalizam incarcarile...")
    storage.close() 
    
    print("\n" + "="*30)
    print("✅ TEST FINALIZAT.")
    print("Verifica acum in dashboard-ul Cloudflare R2 daca fisierul a aparut.")
    print("="*30)

if __name__ == "__main__":
    run_test()

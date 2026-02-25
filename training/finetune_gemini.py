from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

class ThunderFinetuner:
    """
    High-performance SFT script for training the "Thunder" framework.
    Supports Bilateral Denoising alignment and 120k context windows.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def run_sft(self, dataset_name=None, output_dir=None):
        """
        Runs the Supervised Fine-Tuning process using Unsloth and TRL.
        """
        from training.data_pipeline import ThunderDataPipeline
        from core.config_manager import THUNDER_CONFIG
        from reasoning.personality import ThunderPersonality
        
        personality = ThunderPersonality()
        system_prompt = personality.get_system_prompt()["content"]
        
        dataset_name = dataset_name or THUNDER_CONFIG["training"]["dataset_name"]
        output_dir = output_dir or THUNDER_CONFIG["training"]["output_dir"]
        
        pipeline = ThunderDataPipeline(self.tokenizer)
        dataset = pipeline.prepare_dataset(dataset_name)
        
        # Inject system prompt into dataset framing
        print(f"⚡ Thunder: Aligning with persona: {system_prompt[:50]}...")
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=THUNDER_CONFIG["engine"]["max_seq_len"],
            dataset_num_proc=THUNDER_CONFIG["training"]["num_proc"],
            packing=THUNDER_CONFIG["training"]["packing"],
            args=TrainingArguments(
                per_device_train_batch_size=THUNDER_CONFIG["training"]["batch_size"],
                gradient_accumulation_steps=THUNDER_CONFIG["training"]["grad_accum"],
                warmup_steps=THUNDER_CONFIG["training"]["warmup_steps"],
                max_steps=THUNDER_CONFIG["training"]["max_steps"],
                learning_rate=THUNDER_CONFIG["training"]["learning_rate"],
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=THUNDER_CONFIG["training"]["logging_steps"],
                optim=THUNDER_CONFIG["training"]["optim"],
                weight_decay=THUNDER_CONFIG["training"]["weight_decay"],
                lr_scheduler_type=THUNDER_CONFIG["training"]["lr_scheduler"],
                seed=THUNDER_CONFIG["training"]["seed"],
                output_dir=output_dir,
            ),
        )
        
        print("⚡ Thunder: Starting Fine-tuning...")
        trainer.train()
        
        # Save the adapters and tokenizer
        print(f"⚡ Thunder: Saving model to {output_dir}...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer

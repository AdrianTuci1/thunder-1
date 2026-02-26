from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
from training.noise_scheduler import ThunderNoiseScheduler
from training.loss_functions import HybridLoss

class ThunderTrainer(SFTTrainer):
    """
    Custom Trainer for text diffusion.
    Overrides compute_loss to inject noise and use HybridLoss.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_scheduler = ThunderNoiseScheduler()
        self.hybrid_loss_fn = HybridLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Forward pass with noise injection and custom boundary-aware loss.
        """
        input_ids = inputs.get("input_ids")
        # diffusion_step acts as 't' in the schedule [B]
        t = inputs.get("diffusion_step")
        
        # If no diffusion step (fallback), use standard SFT loss
        if t is None:
            return super().compute_loss(model, inputs, return_outputs)

        # 1. Get base embeddings
        embed_module = model.get_input_embeddings()
        embeddings = embed_module(input_ids) # [B, L, D]

        # 2. Sample noise and degrade signal
        # For full-sequence generation, if we don't have explicit prompt boundaries
        # in the input dict, we noise the whole sequence. In a production text-diffusion
        # model, you would typically mask the prompt condition so it remains clean `z_0`,
        # but the model adapter handles `condition` + `latent_field` during inference.
        # Here we train it as an unconditional sequence layout or condition it if we split it.
        noise = torch.randn_like(embeddings)
        noisy_embeddings = self.noise_scheduler.add_noise(embeddings, noise, t)

        # 3. Model forward pass
        # Use inputs_embeds to bypass standard token embedding
        outputs = model(
            inputs_embeds=noisy_embeddings,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 4. Noise Prediction Layer (Adapted Non-Causal Diffusion)
        # Use the adapter to integrate timestep info and predict noise (ε) 
        # from the hidden states of the backbone.
        predicted_noise = model.predict_noise(outputs.hidden_states[-1], t)

        # 5. Calculate Denoising Loss
        # Normalize t to [0, 1] for SNR calculation
        t_norm = t.float() / self.noise_scheduler.num_train_timesteps
        
        loss = self.hybrid_loss_fn.calculate_loss(
            predicted_noise=predicted_noise,
            target_noise=noise,
            timesteps=t_norm
        )

        return (loss, outputs) if return_outputs else loss

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
        Runs the Supervised Fine-Tuning process using Unsloth and ThunderTrainer.
        """
        from training.data_pipeline import ThunderDataPipeline
        from core.config_manager import THUNDER_CONFIG
        from reasoning.personality import ThunderPersonality
        from core.model_adapter import ThunderModelAdapter
        
        personality = ThunderPersonality()
        system_prompt = personality.get_system_prompt()["content"]
        
        dataset_name = dataset_name or THUNDER_CONFIG["training"]["dataset_name"]
        output_dir = output_dir or THUNDER_CONFIG["training"]["output_dir"]

        # 1. Adapt model for Diffusion before training
        adapter = ThunderModelAdapter(self.model)
        self.model = adapter.adapt_for_diffusion()
        
        pipeline = ThunderDataPipeline(self.tokenizer)
        dataset = pipeline.prepare_dataset(dataset_name, augment=True)
        
        # Inject system prompt into dataset framing
        print(f"⚡ Thunder: Aligning with persona: {system_prompt[:50]}...")
        
        # Torch check for RunPod/SSH instance
        try:
            import torch
            has_bf16 = torch.cuda.is_bf16_supported()
        except (ImportError, RuntimeError):
            has_bf16 = False
        
        # Use ThunderTrainer instead of SFTTrainer
        trainer = ThunderTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="input_ids",
            max_seq_length=THUNDER_CONFIG["engine"]["max_seq_len"],
            dataset_num_proc=THUNDER_CONFIG["training"]["num_proc"],
            packing=THUNDER_CONFIG["training"]["packing"],
            args=TrainingArguments(
                per_device_train_batch_size=THUNDER_CONFIG["training"]["batch_size"],
                gradient_accumulation_steps=THUNDER_CONFIG["training"]["grad_accum"],
                warmup_steps=THUNDER_CONFIG["training"]["warmup_steps"],
                max_steps=THUNDER_CONFIG["training"]["max_steps"],
                learning_rate=THUNDER_CONFIG["training"]["learning_rate"],
                fp16=not has_bf16,
                bf16=has_bf16,
                logging_steps=THUNDER_CONFIG["training"]["logging_steps"],
                optim=THUNDER_CONFIG["training"]["optim"],
                weight_decay=THUNDER_CONFIG["training"]["weight_decay"],
                lr_scheduler_type=THUNDER_CONFIG["training"]["lr_scheduler"],
                seed=THUNDER_CONFIG["training"]["seed"],
                output_dir=output_dir,
                save_strategy="steps",
                save_steps=THUNDER_CONFIG["training"]["save_steps"],
                save_total_limit=THUNDER_CONFIG["training"]["save_total_limit"],
                load_best_model_at_end=False,
                # Ensure the trainer doesn't try to remove auxiliary columns we need (diffusion_step)
                remove_unused_columns=False,
            ),
        )
        
        print("⚡ Thunder: Starting Fine-tuning (Diffusion Loop)...")
        trainer.train()
        
        # Save the adapters and tokenizer
        print(f"⚡ Thunder: Saving model to {output_dir}...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer

from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import inspect


def build_lora_model(model, lora_config, target_modules=None,):
    if hasattr(model, "peft_config"):
        raise ValueError(
            "This model already has a PEFT/LoRA adapter. Restart the kernel "
            "and run the notebook from the beginning instead of wrapping it twice."
        )
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    return model




def build_training_args(
    output_dir="./checkpoints",
    batch_size=1,
    eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy = 'epoch',
    save_steps=100,
    eval_strategy="epoch",
    max_grad_norm=1.0,
    dataloader_num_workers=0,
    warmup_steps=0,
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
):

    kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,

        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},

        "fp16": False,
        "bf16": True,

        "optim": "paged_adamw_8bit",

        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": lr_scheduler_type,

        "weight_decay": 0.01,
        # Trainer clips the global gradient norm immediately before each
        # optimizer update (after gradient accumulation).
        "max_grad_norm": max_grad_norm,

        "logging_strategy": "steps",
        "logging_steps": logging_steps,

        "save_total_limit": 3,
        "load_best_model_at_end": load_best_model_at_end,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,

        # Validation only needs the loss. Keeping full vocabulary logits from
        # a vision-language model would consume a large amount of RAM/VRAM.
        "prediction_loss_only": True,

        "report_to": "tensorboard",

        "remove_unused_columns": False,
        "dataloader_num_workers": dataloader_num_workers,

        "seed": 42,
    }
    if save_strategy != 'epoch':
        kwargs['save_strategy'] = save_strategy
        kwargs['save_steps'] = save_steps
    else:
        kwargs['save_strategy'] = save_strategy
        
        
    

    supported = inspect.signature(
        TrainingArguments.__init__
    ).parameters

    # Transformers renamed evaluation_strategy to eval_strategy. Support both
    # names so this helper works with the installed version and older releases.
    if eval_strategy is not None:
        evaluation_key = (
            "eval_strategy"
            if "eval_strategy" in supported
            else "evaluation_strategy"
        )
        kwargs[evaluation_key] = eval_strategy

    kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in supported
    }

    return TrainingArguments(**kwargs)


def build_trainer(
    model,
    train_dataset,
    data_collator,
    training_args,
    eval_dataset=None,
):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    return trainer

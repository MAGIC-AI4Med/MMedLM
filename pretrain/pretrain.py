import transformers
from transformers import Trainer, HfArgumentParser
from dataset import InternlmDataset
from dataclasses import dataclass, field
from typing import Optional
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)

# Define classes for argument parsing
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_path: Optional[str] = field(default="internlm-chat-7b-v1_1")

@dataclass
class DataArguments:
    """
    Arguments for data processing like paths for train, validation datasets etc.
    """
    train_dataset_path: str = field(default=None, metadata={"help": "Path to the training data."})
    val_dataset_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Custom training arguments including model max length and optimization parameters.
    """
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length."})
    gradient_clipping: float = field(default=None)

# Custom Trainer class to handle FSDP saving
class CustomTrainer(Trainer):
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save the model. Overrides Trainer's save_model to handle FSDP.
        """
        if self.fsdp is not None:
            output_dir = output_dir or self.args.output_dir
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state_dict = self.model.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=cpu_state_dict)
            if self.args.push_to_hub and not _internal_call:
                self.push_to_hub(commit_message="Model save")
        else:
            super().save_model(output_dir, _internal_call)

# Utility function to handle safe saving of models for Hugging Face trainers
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    Safely saves the model state for Hugging Face trainers with custom handling for FSDP.
    """
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        trainer._save(output_dir, state_dict=cpu_state_dict)

# Main function to set up and train the model
def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set up data
    print("Setup Data")
    train_dataset = InternlmDataset(data_args.train_dataset_path)
    eval_dataset = InternlmDataset(data_args.val_dataset_path)

    # Set up model
    print("Setup Model")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True
    )
    
    # Initialize trainer
    trainer = CustomTrainer(
        model=model, 
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # Train the model
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    main()

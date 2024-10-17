import os
import time
from tqdm import tqdm
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, RandomSampler
from typing import Callable, Optional, Dict, List, Tuple, Union, Any
from dataclasses import asdict, dataclass, field, fields
from packaging import version
import transformers
from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)
from transformers.deepspeed import deepspeed_init
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments

if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import (
        PeftModel,
        set_peft_model_state_dict,
    )

from models.lavit_for_pigeon import LaVITforPigeon, MutualMaskGenerator
from models.trainer.optimization import get_scheduler
from models.trainer.trainer_callback import CustomProgressCallback
from data_utils import PigeonDataset, PigeonCollator

logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

@torch.no_grad()
def gather_together(data):
    dist.barrier()
    world_size = dist.get_world_size()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)
    return gather_data

@dataclass
class PigeonTrainingArguments(TrainingArguments):
    eval_num: int = field(default=1000, metadata={"help": ("Samples num for evaluation")})
    hist_mask_ratio: float = field(default=0.2, metadata={
        "help": "The mask ratio of user interaction history. Invalid if the mask type is random."})
    min_lr_ratio: float = field(
        default=1e-6, metadata={"help": "The min lr ratio reqpect to the learning rate, only used to cosine lr scheduler."})
    add_special: bool = field(
        default=True, metadata={"help": "Whether add image start token and end token into the model input or not."}
    )

class PigeonTrainer(transformers.Trainer):
    """
    Customed Trainer for pigeon.
    """ 
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: PigeonTrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[PigeonDataset] = None,
        eval_dataset: Optional[Union[PigeonDataset, Dict[str, PigeonDataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        # Check if args is of the correct type
        if args is not None and not isinstance(args, PigeonTrainingArguments):
            raise ValueError(f"Expected `args` to be of type `CustomTrainingArguments`, but got {type(args)}")
        
        # Call the parent class's __init__ method
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if not self.args.disable_tqdm:
            self.remove_callback(ProgressCallback)
            self.add_callback(CustomProgressCallback)

        # supress the text tokens
        supress_range = range(3, 32000)
        supress_tokens = [x for x in supress_range]
        self.generation_config = transformers.GenerationConfig(
            top_p=1.0,
            temperature=1.0,
            num_beams=2,
            min_new_tokens=20,
            max_new_tokens=300,
            suppress_tokens=supress_tokens,
            bos_token_id=32000,
            eos_token_id=32001,
            pad_token_id=model.llama_tokenizer.pad_token_id,
        )

        self.args = args
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    
    def get_eval_dataloader(self, eval_dataset: Optional[PigeonDataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataset = eval_dataset.shuffle().select(range(self.args.eval_num))  # 采样验证

        data_collator = self.data_collator

        # if is_datasets_available() and isinstance(eval_dataset, PigeonDataset):
        #     eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        # else:
        #     data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

    def get_test_dataloader(self, test_dataset: PigeonDataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        data_collator = self.data_collator

        # if is_datasets_available() and isinstance(test_dataset, PigeonDataset):
        #     test_dataset = self._remove_unused_columns(test_dataset, description="test")
        # else:
        #     data_collator = self._get_collator_with_removed_columns(data_collator, description="test")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(test_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(test_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        # We use the same batch_size as for eval.
        return self.accelerator.prepare(DataLoader(test_dataset, **dataloader_params))

    def compute_loss(self, model, inputs, return_outputs=False, return_labels=False, training=True, use_cache=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs, labels = model(
            inputs, self.args.hist_mask_ratio, 
            self.args.add_special, training=training, label_smooth_factor=self.args.label_smoothing_factor,
            use_cache=use_cache,
        )
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        if self.args.label_smoothing_factor != 0:
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        # if torch.isnan(loss).any():
        #     for name, param in model.named_parameters():
        #         if torch.isnan(param).any():
        #             logger.info(f"NaN detected in {name}")  # 模型参数中不存在nan

        #     # logger.info(f"inputs:{inputs}")
        #     # logger.info(f"outputs:{outputs}")
        #     # logger.info(f"labels:{labels}")
        #     logger.info(f"loss:{loss}")
        #     # raise ValueError(f"Error! Loss is NAN.")

        if return_outputs and return_labels:
            return (loss, outputs, labels)
        elif return_outputs:
            return (loss, outputs)
        else:
            return loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs, labels = self.compute_loss(model, inputs, return_outputs=True,
                                                          return_labels=True, training=False, use_cache=False,)
            loss = loss.mean().detach()

            ##########
            if torch.isnan(loss).any():
                for name, param in model.named_parameters():
                    if torch.isnan(param).any():
                        logger.info(f"NaN detected in {name}")  # 模型参数中不存在nan

                # logger.info(f"inputs:{inputs}")
                # logger.info(f"outputs:{outputs}")
                # logger.info(f"labels:{labels}")
                logger.info(f"loss:{loss}")
                # raise ValueError(f"Error! Loss is NAN.")

            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
            else:
                logits = outputs[1:]
        
        if prediction_loss_only:
            return (loss, None, None)
        
        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
        
        return (loss, logits, labels)

    def print_memory_usage(self):
        memory_allocated = torch.cuda.memory_allocated(self.accelerator.device)
        memory_reserved = torch.cuda.memory_reserved(self.accelerator.device)
        logger.info(f"Memory allocated: {memory_allocated / (1024 ** 3):.2f} GB")
        logger.info(f"Memory reserved: {memory_reserved / (1024 ** 3):.2f} GB")
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        logger.info(f"optimizer_type-{type(self.optimizer)}, optimizer-{self.optimizer}")
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(self.args.lr_scheduler_type,
                                              optimizer=self.optimizer if optimizer is None else optimizer,
                                              num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                                              num_training_steps=num_training_steps,
                                              min_lr_ratio=self.args.min_lr_ratio)
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # Remaining the state_dict argument to be compatible with original trainer class
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_class = LaVITforPigeon
        if not isinstance(self.model, supported_class):
            unwrapped_model = unwrap_model(self.model)
            if not isinstance(unwrapped_model, supported_class):
                raise ValueError(f"Invalid model class: {unwrapped_model.__class__.__name__}")
        else:
            unwrapped_model = self.model
        
        if unwrapped_model.mask_generator is not None:
            mask_generator_folder = os.path.join(output_dir, "mask_generator")
            os.makedirs(mask_generator_folder, exist_ok=True)

            mask_generator_state_dict = unwrapped_model.mask_generator.state_dict()
            if self.args.save_safetensors:
                safetensors.torch.save_file(mask_generator_state_dict, os.path.join(mask_generator_folder, SAFE_WEIGHTS_NAME))
            else:
                torch.save(mask_generator_state_dict, os.path.join(output_dir, "mask_generator", WEIGHTS_NAME))

        adapter_folder = os.path.join(output_dir, "adapter")
        os.makedirs(adapter_folder, exist_ok=True)

        adapter_state_dict = unwrapped_model.adapter.state_dict()
        if self.args.save_safetensors:
            safetensors.torch.save_file(adapter_state_dict, os.path.join(adapter_folder, SAFE_WEIGHTS_NAME))
        else:
            torch.save(adapter_state_dict, os.path.join(output_dir, "adapter", WEIGHTS_NAME))

        llama_folder = os.path.join(output_dir, "llama")
        os.makedirs(llama_folder, exist_ok=True)
        
        unwrapped_model.llama_model.save_pretrained(
            llama_folder, state_dict=None, safe_serialization=self.args.save_safetensors
        )

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
            model = self.model

        if not isinstance(model, LaVITforPigeon):
            raise ValueError(f"Invalid model class: {model.__class__.__name__}")
        
        mask_generator_weights_file = os.path.join(resume_from_checkpoint, "mask_generator", WEIGHTS_NAME)
        safe_mask_generator_weights_file = os.path.join(resume_from_checkpoint, "mask_generator", SAFE_WEIGHTS_NAME)
        adapter_weights_file = os.path.join(resume_from_checkpoint, "adapter", WEIGHTS_NAME)
        safe_adapter_weights_file = os.path.join(resume_from_checkpoint, "adapter", SAFE_WEIGHTS_NAME)
        llama_adapter_weights_file = os.path.join(resume_from_checkpoint, "llama", ADAPTER_WEIGHTS_NAME)
        safe_llama_adapter_weights_file = os.path.join(resume_from_checkpoint, "llama", ADAPTER_SAFE_WEIGHTS_NAME)

        if not any(
            os.path.isfile(f)
            for f in [
                mask_generator_weights_file,
                safe_mask_generator_weights_file,
                llama_adapter_weights_file,
                safe_llama_adapter_weights_file,
                adapter_weights_file,
                safe_adapter_weights_file,
            ]
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if model.mask_generator is not None:
            if os.path.isfile(mask_generator_weights_file) or os.path.isfile(safe_mask_generator_weights_file):
                # We load the model state dict on the CPU to avoid an OOM error.
                if self.args.save_safetensors and os.path.isfile(safe_mask_generator_weights_file):
                    state_dict = safetensors.torch.load_file(safe_mask_generator_weights_file)  # , device="cpu")
                else:
                    state_dict = torch.load(mask_generator_weights_file)  # , map_location="cpu")

                # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                # which takes *args instead of **kwargs
                model.mask_generator.load_state_dict(state_dict, False)
                # release memory
                del state_dict
        
        if os.path.isfile(adapter_weights_file) or os.path.isfile(safe_adapter_weights_file):
            # We load the model state dict on the CPU to avoid an OOM error.
            if self.args.save_safetensors and os.path.isfile(safe_adapter_weights_file):
                state_dict = safetensors.torch.load_file(safe_adapter_weights_file)  # , device="cpu")
            else:
                state_dict = torch.load(adapter_weights_file)  # , map_location="cpu")

            # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
            # which takes *args instead of **kwargs
            model.adapter.load_state_dict(state_dict, False)
            # release memory
            del state_dict

        # Load adapters
        if is_peft_available() and isinstance(model.llama_model, PeftModel):
            llama_adapter_path = os.path.join(resume_from_checkpoint, "llama")
            # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
            if hasattr(model.llama_model, "active_adapter") and hasattr(model.llama_model, "load_adapter"):
                if os.path.exists(llama_adapter_path):
                    model.llama_model.load_adapter(llama_adapter_path, model.llama_model.active_adapter, is_trainable=True)
                else:
                    logger.warning(
                        "The intermediate checkpoints of PEFT may not be saved correctly, "
                        f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                        "Check some examples here: https://github.com/huggingface/peft/issues/96"
                    )
            else:
                logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")

    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        best_mask_generator_path = os.path.join(self.state.best_model_checkpoint, "mask_generator", WEIGHTS_NAME)
        best_safe_mask_generator_path = os.path.join(self.state.best_model_checkpoint, "mask_generator", SAFE_WEIGHTS_NAME)
        best_adapter_path = os.path.join(self.state.best_model_checkpoint, "adapter", WEIGHTS_NAME)
        best_safe_adapter_path = os.path.join(self.state.best_model_checkpoint, "adapter", SAFE_WEIGHTS_NAME)
        best_llama_adapter_model_path = os.path.join(self.state.best_model_checkpoint, "llama", ADAPTER_WEIGHTS_NAME)
        best_safe_llama_adapter_model_path = os.path.join(self.state.best_model_checkpoint, "llama", ADAPTER_SAFE_WEIGHTS_NAME)

        model = self.model
        if not isinstance(model, LaVITforPigeon):
            raise ValueError(f"Invalid model class: {model.__class__.__name__}")
        
        if (
            (os.path.exists(best_mask_generator_path)
            and os.path.exists(best_llama_adapter_model_path)
            and os.path.exists(best_adapter_path))
            or (os.path.exists(best_safe_mask_generator_path)
            and os.path.exists(best_safe_llama_adapter_model_path)
            and os.path.exists(best_safe_adapter_path))
        ):
            if model.mask_generator is not None:
                if os.path.isfile(best_mask_generator_path) or os.path.isfile(best_safe_mask_generator_path):
                    # We load the model state dict on the CPU to avoid an OOM error.
                    if self.args.save_safetensors and os.path.isfile(best_safe_mask_generator_path):
                        state_dict = safetensors.torch.load_file(best_safe_mask_generator_path)  # , device="cpu")
                    else:
                        state_dict = torch.load(best_mask_generator_path)  # , map_location="cpu")

                    # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                    # which takes *args instead of **kwargs
                    model.mask_generator.load_state_dict(state_dict, False)
                    # release memory
                    del state_dict
            
            if os.path.isfile(best_adapter_path) or os.path.isfile(best_safe_adapter_path):
                # We load the model state dict on the CPU to avoid an OOM error.
                if self.args.save_safetensors and os.path.isfile(best_safe_adapter_path):
                    state_dict = safetensors.torch.load_file(best_safe_adapter_path)  # , device="cpu")
                else:
                    state_dict = torch.load(best_adapter_path)  # , map_location="cpu")

                # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                # which takes *args instead of **kwargs
                model.adapter.load_state_dict(state_dict, False)
                # release memory
                del state_dict

            if is_peft_available() and isinstance(model.llama_model, PeftModel):
                # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
                if hasattr(model.llama_model, "active_adapter") and hasattr(model.llama_model, "load_adapter"):
                    best_llama_adapter_path = os.path.join(self.state.best_model_checkpoint, "llama")
                    if os.path.exists(best_llama_adapter_model_path) or os.path.exists(best_safe_llama_adapter_model_path):
                        model.llama_model.load_adapter(best_llama_adapter_path, model.llama_model.active_adapter)
                    else:
                        raise ValueError(
                            "The intermediate checkpoints of PEFT may not be saved correctly, "
                            f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                            "Check some examples here: https://github.com/huggingface/peft/issues/96"
                        )
                else:
                    raise ValueError("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
        else:
            raise ValueError(
                f"Could not locate the best model at {self.state.best_model_checkpoint}"
            )

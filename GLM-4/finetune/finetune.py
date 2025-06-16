# -*- coding: utf-8 -*-
import dataclasses as dc
import functools
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Optional, Union

import jieba
import numpy as np
import ruamel.yaml as yaml
import torch
import typer
from datasets import Dataset, DatasetDict, NamedSplit, Split, load_dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import PeftConfig, get_peft_config, get_peft_model
from rouge_chinese import Rouge
from torch import nn
from torch.distributed.tensor.parallel import loss_parallel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    GenerationConfig,
    PreTrainedTokenizer,
    Seq2SeqTrainingArguments,
)
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer as _Seq2SeqTrainer
import torch.nn.functional as F
import json


# For Ascend NPU, please add this
# import torch_npu
# from torch_npu.contrib import transfer_to_npu

app = typer.Typer(pretty_exceptions_show_locals=False)

# Enable Flash Attention globally
torch.backends.cuda.enable_flash_sdp(True)


class TorchSaveDataset(Dataset):
    def __init__(self, data_dir, file_paths):
        self.data_dir = data_dir
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        input_ids = []
        labels = []
        label_indices = []
        lm_sum = []
        for id in idx:
            input_ids.append(torch.load(self.data_dir + self.file_paths[id]['input_ids']))
            labels.append(torch.load(self.data_dir + self.file_paths[id]['labels']))
            label_indices.append(torch.load(self.data_dir + self.file_paths[id]['label_indices']))
            lm_sum.append(torch.load(self.data_dir + self.file_paths[id]['lm_sum']))

        return {'input_ids': input_ids, 'labels': labels, 'label_indices': label_indices, 'lm_sum': lm_sum}



class NPZDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        input_ids = []
        labels = []
        output_ids = []
        for id in idx:
            temp = np.load(self.file_paths[id])
            input_ids.append(temp['input_ids'])
            if 'labels' in temp:
                labels.append(temp['labels'])
            else:
                output_ids.append(temp['output_ids'])

        if len(labels) > 0:
            return {'input_ids': input_ids, 'labels': labels}
        else:
            return {'input_ids': input_ids, 'output_ids': output_ids}


class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        # Torch Save Dataset
        if 'lm_sum' in features[0]:
            batch = {}
            input_ids = []
            labels = []
            label_indices = []
            lm_sum = []
            seq_sum = []

            for feature in features:
                seq_sum.append(len(feature['input_ids']))
            longest_length = max(seq_sum)

            for feature in features:
                temp = torch.full((longest_length,), fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
                temp[:len(feature['input_ids'])] = feature['input_ids']
                input_ids.append(temp)

                temp = torch.full((longest_length,8), fill_value=-100, dtype=torch.bfloat16)
                temp[:len(feature['labels'])] = feature['labels']
                labels.append(temp)

                temp = torch.full((longest_length,8), fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
                temp[:len(feature['label_indices'])] = feature['label_indices']
                label_indices.append(temp)

                lm_sum.append(feature['lm_sum'])

            input_ids = torch.stack(input_ids)
            batch['input_ids'] = input_ids

            labels = torch.stack(labels)
            label_indices = torch.stack(label_indices)

            # TODO: Possible to set 151552 dynamically?
            temp_labels = torch.full((labels.shape[0], labels.shape[1], 151552), fill_value=-100, dtype=torch.bfloat16)
            temp_labels = temp_labels.scatter(2, label_indices, labels)
            batch['labels'] = temp_labels

            seq_sum = torch.tensor(np.sum(seq_sum), dtype=torch.bfloat16)
            batch['seq_sum'] = seq_sum
            torch.argmax(temp_labels[0], dim=1)[15:]
            batch['tokenizer'] = self.tokenizer

            return [batch]

        # NPZ Dataset
        if 'labels' in features[0]:
            if type(features[0]['labels']) is np.ndarray:
                batch = {}
                input_ids = []
                labels = []
                for feature in features:
                    input_ids.append(feature['input_ids'])
                    labels.append(feature['labels'])

                batch['input_ids'] = torch.tensor(np.stack(input_ids))
                labels = torch.tensor(np.stack(labels), dtype=torch.bfloat16)
                batch['labels'] = labels
                batch['mask_sum'] = torch.sum((labels != -100))

                return [batch]

        # Standard Dataset
        output_ids = [feature["output_ids"] for feature in features] if "output_ids" in features[0].keys() else None
        if output_ids is not None:
            max_output_length = max(len(out) for out in output_ids)
            if self.pad_to_multiple_of is not None:
                max_output_length = (
                    (max_output_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (max_output_length - len(feature["output_ids"]))
                if isinstance(feature["output_ids"], list):
                    feature["output_ids"] = feature["output_ids"] + remainder
                else:
                    feature["output_ids"] = np.concatenate([feature["output_ids"], remainder]).astype(np.int64)
        return super().__call__(features, return_tensors)


class Seq2SeqTrainer(_Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override loss calculation to use token logprobs when available.
        Falls back to standard cross-entropy when not in logprob_mode.
        """
        if hasattr(self.args, 'logprob_mode'):
            if self.args.logprob_mode:
                accum_batch_size = torch.tensor(self.args.gradient_accumulation_steps, dtype=torch.bfloat16)

                # Reshape inputs for model
                outputs = model(input_ids=inputs[0]["input_ids"])

                # Mask inputs before calc. loss fn.
                mask = torch.unsqueeze(torch.max(inputs[0]["labels"] > -100, dim=2)[0], dim=2)

                # Get model's log probabilities for each token
                # logits = outputs.logits.to(dtype=torch.bfloat16)
                log_probs = torch.nn.functional.log_softmax(outputs.logits.to(dtype=torch.bfloat16), dim=2)

                # Use precomputed logprobs from input data
                labels_softmax = torch.nn.functional.softmax(inputs[0]["labels"], dim=2)

                # print(torch.argmax(labels_softmax, dim=2)[0])
                # print(torch.argmax(log_probs, dim=2)[0])
                # print()
                #
                # print(torch.argmax(labels_softmax, dim=2)[1])
                # print(torch.argmax(log_probs, dim=2)[1])
                # print()
                #
                # print(torch.argmax(labels_softmax, dim=2)[2])
                # print(torch.argmax(log_probs, dim=2)[2])
                # print()

                log_probs = log_probs * mask
                labels_softmax = labels_softmax * mask

                # Calculate loss
                # loss = F.cross_entropy(input=logits, target=labels.exp(), reduction="none")
                loss = F.kl_div(log_probs, labels_softmax, reduction="none", log_target=False)
                loss = loss.sum() / (inputs[0]['seq_sum'] * accum_batch_size) # batchmean kl_div, hf trainer can't count logprob batches

                return (loss, outputs) if return_outputs else loss

            else:
                # Standard cross-entropy loss
                return super().compute_loss(model, inputs, return_outputs)
        else:
            # Standard cross-entropy loss
            return super().compute_loss(model, inputs, return_outputs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys=None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():  # Ensure no gradient computation
            if self.args.predict_with_generate:
                output_ids = inputs.pop("output_ids")
            input_ids = inputs["input_ids"]

            del inputs["labels"]
            loss, generated_tokens, labels = super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
            )

            generated_tokens = generated_tokens[:, input_ids.size()[1] :]
            labels = output_ids

            del inputs, input_ids, output_ids
            torch.cuda.empty_cache()

        return loss, generated_tokens, labels


@dc.dataclass
class DataConfig(object):
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    num_proc: Optional[int] = None
    logprob_mode: bool = False  # Flag to enable logprob training

    @property
    def data_format(self) -> str:
        return Path(self.train_file).suffix

    @property
    def data_files(self) -> dict[NamedSplit, str]:
        return {
            split: data_file
            for split, data_file in zip(
                [Split.TRAIN, Split.VALIDATION, Split.TEST],
                [self.train_file, self.val_file, self.test_file],
            )
            if data_file is not None
        }


@dc.dataclass
class FinetuningConfig(object):
    data_config: DataConfig

    max_input_length: int
    max_output_length: int
    combine: bool
    freezeV: bool
    logprob_mode: bool = False  # Main logprob mode flag

    training_args: Seq2SeqTrainingArguments = dc.field(
        default_factory=lambda: Seq2SeqTrainingArguments(output_dir="./output")
    )
    peft_config: Optional[PeftConfig] = None
    swanlab: Optional[str] = "cloud"

    def __post_init__(self):
        if not self.training_args.do_eval or self.data_config.val_file is None:
            self.training_args.do_eval = False
            self.training_args.evaluation_strategy = "no"
            self.data_config.val_file = None
        else:
            self.training_args.per_device_eval_batch_size = (
                self.training_args.per_device_eval_batch_size or self.training_args.per_device_train_batch_size
            )
        if self.swanlab != "disabled":
            os.environ["SWANLAB_PROJ_NAME"] = "GLM4-Finetune"
        if self.swanlab == "local":
            os.environ["SWANLAB_MODE"] = "local"

        # Propagate logprob_mode to training args for loss calculation
        self.training_args.logprob_mode = self.logprob_mode

    @classmethod
    def from_dict(cls, **kwargs) -> "FinetuningConfig":
        training_args = kwargs.get("training_args", None)
        if training_args is not None and not isinstance(training_args, Seq2SeqTrainingArguments):
            gen_config = training_args.get("generation_config")
            if not isinstance(gen_config, GenerationConfig):
                training_args["generation_config"] = GenerationConfig(**gen_config)
            kwargs["training_args"] = Seq2SeqTrainingArguments(**training_args)

        data_config = kwargs.get("data_config")
        if not isinstance(data_config, DataConfig):
            kwargs["data_config"] = DataConfig(**data_config)

        peft_config = kwargs.get("peft_config", None)
        if peft_config is not None and not isinstance(peft_config, PeftConfig):
            kwargs["peft_config"] = get_peft_config(config_dict=peft_config)
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "FinetuningConfig":
        path = Path(path)
        parser = yaml.YAML(typ="safe", pure=True)
        parser.indent(mapping=2, offset=2, sequence=4)
        parser.default_flow_style = False
        kwargs = parser.load(path)
        return cls.from_dict(**kwargs)


def _load_datasets(
    data_dir: str,
    data_format: str,
    data_files: dict[NamedSplit, str],
    num_proc: Optional[int],
) -> DatasetDict:
    if data_format == ".jsonl":
        dataset_dct = load_dataset(
            data_dir,
            data_files=data_files,
            split=None,
            num_proc=num_proc,
        )
    else:
        raise NotImplementedError(f"Cannot load dataset in the '{data_format}' format.")
    return dataset_dct


class DataManager(object):
    def __init__(self, data_dir: str, data_config: DataConfig):
        self._num_proc = data_config.num_proc

        self._dataset_dct = _load_datasets(
            data_dir,
            data_config.data_format,
            data_config.data_files,
            self._num_proc,
        )

    def _get_dataset(self, split: NamedSplit) -> Optional[Dataset]:
        return self._dataset_dct.get(split, None)

    def get_dataset(
        self,
        split: NamedSplit,
        process_fn: Callable[[dict[str, Any]], dict[str, Any]],
        batched: bool = True,
        remove_orig_columns: bool = True,
    ) -> Optional[Dataset]:
        orig_dataset = self._get_dataset(split)
        if orig_dataset is None:
            return

        if remove_orig_columns:
            remove_columns = orig_dataset.column_names
        else:
            remove_columns = None
        return orig_dataset.map(
            process_fn,
            batched=batched,
            remove_columns=remove_columns,
            num_proc=self._num_proc,
        )


def process_message(message):
    if "tools" in message and message["role"] == "system":
        for tool in message["tools"]:
            parameters = tool["function"]["parameters"]["properties"]
            tool["function"]["parameters"]["properties"] = {k: v for k, v in parameters.items() if v is not None}
    elif "tools" in message:
        del message["tools"]
    return message


def process_batch(
    batch: Mapping[str, Sequence],
    tokenizer: PreTrainedTokenizer,
    max_input_length: int,
    max_output_length: int,
    combine: bool,
    logprob_mode: bool = False
) -> dict[str, list]:
    batched_conv = batch["messages"]
    batched_input_ids = []
    batched_labels = []
    for conv in batched_conv:
        input_ids = [151331, 151333]
        loss_masks = [False, False]
        if combine:
            new_input_ids = tokenizer.apply_chat_template(conv, tokenize=True, return_dict=False)
            input_ids = new_input_ids
            loss_masks = [False] * len(input_ids)
            last_assistant_index = len(input_ids) - input_ids[::-1].index(151337) - 1
            for j in range(last_assistant_index + 1, len(input_ids)):
                loss_masks[j] = True
        else:
            for message in conv:
                message = process_message(message)
                loss_mask_val = False if message["role"] in ("system", "user", "observation") else True
                new_input_ids = tokenizer.apply_chat_template([message], tokenize=True, return_dict=False)[2:]
                input_ids += new_input_ids
                loss_masks += [loss_mask_val] * len(new_input_ids)

        input_ids.append(151336)  # EOS for chat
        loss_masks = [False, *loss_masks]
        labels = []
        for input_id, mask in zip(input_ids, loss_masks):
            if mask:
                labels.append(input_id)
            else:
                labels.append(-100)
        max_length = max_input_length + max_output_length + 1
        batched_input_ids.append(input_ids[:max_length])
        batched_labels.append(labels[:max_length])

    del batched_conv, conv, input_ids, loss_masks, new_input_ids, labels
    torch.cuda.empty_cache()

    return {"input_ids": batched_input_ids, "labels": batched_labels}


def process_batch_eval(
    batch: Mapping[str, Sequence],
    tokenizer: PreTrainedTokenizer,
    max_input_length: int,
    max_output_length: int,
    combine: bool,
) -> dict[str, list]:
    batched_conv = batch["messages"]
    batched_input_ids = []
    batched_output_ids = []

    for conv in batched_conv:
        if combine:
            new_input_ids = tokenizer.apply_chat_template(conv, tokenize=True, return_dict=False)
            input_ids = new_input_ids
            last_assistant_index = len(input_ids) - input_ids[::-1].index(151337) - 1
            output_ids = input_ids[last_assistant_index+1:]
            output_ids.append(151336)

            batched_input_ids.append(input_ids[:last_assistant_index+1])
            batched_output_ids.append(output_ids[:max_output_length])
        else:
            input_ids = [151331, 151333]
            for message in conv:
                if len(input_ids) >= max_input_length:
                    break
                else:
                    message = process_message(message)
                    new_input_ids = tokenizer.apply_chat_template([message], tokenize=True, return_dict=False)[2:]
                    if message["role"] == "assistant":
                        output_prompt, output_ids = (
                            new_input_ids[:1],
                            new_input_ids[1:],
                        )
                        output_ids.append(151336)
                        batched_input_ids.append(input_ids[:max_input_length] + output_prompt[:1])
                        batched_output_ids.append(output_ids[:max_output_length])
                    input_ids += new_input_ids

    del batched_conv, conv, input_ids, new_input_ids, output_ids
    torch.cuda.empty_cache()

    return {"input_ids": batched_input_ids, "output_ids": batched_output_ids}


def load_tokenizer_and_model(
    model_dir: str,
    peft_config: Optional[PeftConfig] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left", trust_remote_code=True)
    if peft_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            use_cache=False,
            torch_dtype=torch.bfloat16,  # Must use BFloat 16
            attn_implementation="flash_attention_2"
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            use_cache=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
    return tokenizer, model


def compute_metrics(eval_preds: EvalPrediction, tokenizer):
    batched_pred_ids, batched_label_ids = eval_preds
    batched_pred_ids[batched_pred_ids == -100] = tokenizer.pad_token_id
    batched_label_ids[batched_label_ids == -100] = tokenizer.pad_token_id
    metrics_dct = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
    for pred_ids, label_ids in zip(batched_pred_ids, batched_label_ids):
        # Strip pad tokens to get accurate eval scores
        pred_ids[pred_ids == tokenizer.pad_token_id] = 198
        label_ids[label_ids == tokenizer.pad_token_id] = 198
        pred_txt = tokenizer.decode(pred_ids).strip()
        label_txt = tokenizer.decode(label_ids).strip()
        print(label_txt)
        print(pred_txt)
        print()
        pred_tokens = list(jieba.cut(pred_txt))
        label_tokens = list(jieba.cut(label_txt))
        rouge = Rouge()
        scores = rouge.get_scores(" ".join(pred_tokens), " ".join(label_tokens))
        for k, v in scores[0].items():
            metrics_dct[k].append(round(v["f"] * 100, 4))
        metrics_dct["bleu-4"].append(
            sentence_bleu(
                [label_tokens],
                pred_tokens,
                smoothing_function=SmoothingFunction().method3,
            )
        )
    return {k: np.mean(v) for k, v in metrics_dct.items()}


@app.command()
def main(
    data_dir: Annotated[str, typer.Argument(help="")],
    model_dir: Annotated[
        str,
        typer.Argument(
            help="A string that specifies the model id of a pretrained model configuration hosted on huggingface.co, or a path to a directory containing a model configuration file."
        ),
    ],
    config_file: Annotated[str, typer.Argument(help="")],
    auto_resume_from_checkpoint: str = typer.Argument(
        default="",
        help="If entered as yes, automatically use the latest save checkpoint. If it is a numerical example 12 15, use the corresponding save checkpoint. If the input is no, restart training",
    ),
    logprob_mode: bool = typer.Option(
        False,
        "--logprob-mode",
        help="Enable training with token log probabilities from preprocess_logprobs.py"
    ),
):
    ft_config = FinetuningConfig.from_file(config_file)
    ft_config.logprob_mode = logprob_mode
    ft_config.data_config.logprob_mode = logprob_mode

    tokenizer, model = load_tokenizer_and_model(model_dir, peft_config=ft_config.peft_config)

    if logprob_mode:
        train_file_paths = []
        val_file_paths = []
        test_file_paths = []
        with open(data_dir + ft_config.data_config.data_files['train'], 'r') as f:
            for line in f:
                train_file_paths.append(json.loads(line))
        with open(data_dir + ft_config.data_config.data_files['validation'], 'r') as f:
            for line in f:
                val_file_paths.append(data_dir + line.strip())
        with open(data_dir + ft_config.data_config.data_files['test'], 'r') as f:
            for line in f:
                test_file_paths.append(data_dir + line.strip())
        train_dataset = TorchSaveDataset(data_dir, train_file_paths)

        # A little hacky but it allows us to run evals on standard completions as opposed to torch save format
        modified_data_config = ft_config.data_config
        modified_data_config.data_files['train'] = 'standard.jsonl'
        modified_data_config.train_file = 'standard.jsonl'
        data_manager = DataManager(data_dir, modified_data_config)

        val_dataset = data_manager.get_dataset(
            Split.VALIDATION,
            functools.partial(
                process_batch_eval,
                tokenizer=tokenizer,
                combine=ft_config.combine,
                max_input_length=ft_config.max_input_length,
                max_output_length=ft_config.max_output_length,
            ),
            batched=True,
        )
        test_dataset = data_manager.get_dataset(
            Split.TEST,
            functools.partial(
                process_batch_eval,
                tokenizer=tokenizer,
                combine=ft_config.combine,
                max_input_length=ft_config.max_input_length,
                max_output_length=ft_config.max_output_length,
            ),
            batched=True,
        )

    else:
        data_manager = DataManager(data_dir, ft_config.data_config)

        train_dataset = data_manager.get_dataset(
            Split.TRAIN,
            functools.partial(
                process_batch,
                tokenizer=tokenizer,
                combine=ft_config.combine,
                max_input_length=ft_config.max_input_length,
                max_output_length=ft_config.max_output_length,
                logprob_mode=ft_config.logprob_mode
            ),
            batched=True,
        )
        val_dataset = data_manager.get_dataset(
            Split.VALIDATION,
            functools.partial(
                process_batch_eval,
                tokenizer=tokenizer,
                combine=ft_config.combine,
                max_input_length=ft_config.max_input_length,
                max_output_length=ft_config.max_output_length,
            ),
            batched=True,
        )
        test_dataset = data_manager.get_dataset(
            Split.TEST,
            functools.partial(
                process_batch_eval,
                tokenizer=tokenizer,
                combine=ft_config.combine,
                max_input_length=ft_config.max_input_length,
                max_output_length=ft_config.max_output_length,
            ),
            batched=True,
        )
        print("train_dataset:", train_dataset)
        if val_dataset is not None:
            print("val_dataset:", val_dataset)
        if test_dataset is not None:
            print("test_dataset:", test_dataset)


    ft_config.training_args.generation_config.pad_token_id = 151329
    ft_config.training_args.generation_config.eos_token_id = [151329, 151336, 151338]
    ft_config.training_args.logprob_mode = ft_config.logprob_mode

    trainer = Seq2SeqTrainer(
        model=model,
        args=ft_config.training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding="longest",
            return_tensors="pt",
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer),
    )

    if auto_resume_from_checkpoint.upper() == "" or auto_resume_from_checkpoint is None:
        trainer.train()
    else:
        output_dir = ft_config.training_args.output_dir
        dirlist = os.listdir(output_dir)
        checkpoint_sn = 0
        for checkpoint_str in dirlist:
            if checkpoint_str.find("eckpoint") > 0 and checkpoint_str.find("tmp") == -1:
                checkpoint = int(checkpoint_str.replace("checkpoint-", ""))
                if checkpoint > checkpoint_sn:
                    checkpoint_sn = checkpoint
        if auto_resume_from_checkpoint.upper() == "YES":
            if checkpoint_sn > 0:
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
                checkpoint_directory = os.path.join(output_dir, "checkpoint-" + str(checkpoint_sn))
                print("resume checkpoint from checkpoint-" + str(checkpoint_sn))
                trainer.train(resume_from_checkpoint=checkpoint_directory)
            else:
                trainer.train()
        else:
            if auto_resume_from_checkpoint.isdigit():
                if int(auto_resume_from_checkpoint) > 0:
                    checkpoint_sn = int(auto_resume_from_checkpoint)
                    model.gradient_checkpointing_enable()
                    model.enable_input_require_grads()
                    checkpoint_directory = os.path.join(output_dir, "checkpoint-" + str(checkpoint_sn))
                    print("resume checkpoint from checkpoint-" + str(checkpoint_sn))
                    trainer.train(resume_from_checkpoint=checkpoint_directory)
            else:
                print(
                    auto_resume_from_checkpoint,
                    "The specified checkpoint sn("
                    + auto_resume_from_checkpoint
                    + ") has not been saved. Please search for the correct checkpoint in the model output directory",
                )

    if test_dataset is not None:
        trainer.predict(test_dataset)


if __name__ == "__main__":
    app()

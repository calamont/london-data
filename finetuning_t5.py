import logging
import os
import json
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
import numpy as np
from datasets import load_dataset, load_metric
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollator,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    local_files: bool = field(
        default=False,
        metadata={"help": "Whether to use local files instead of downloading from s3."},
    )
    sagemaker: Optional[bool] = field(
        default=False, metadata={"help": "Whether running script in AWS Sagemaker"}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv file containing the training data."}
    )
    predict_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv file containing the data for prediction."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )


@dataclass
class T2TDataCollator:
    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = np.vstack([example["input_ids"] for example in batch])
        lm_labels = np.vstack([example["target_ids"] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        attention_mask = np.vstack([example["attention_mask"] for example in batch])
        decoder_attention_mask = np.vstack(
            [example["target_attention_mask"] for example in batch]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lm_labels": lm_labels,
            "decoder_attention_mask": decoder_attention_mask,
        }


def main():

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # If running script on Sagemaker
    if data_args.sagemaker:
        training_args.output_dir = os.environ['SM_OUTPUT_DATA_DIR']
        if training_args.do_train:
            training_args.train_file = os.environ['SM_CHANNEL_TRAIN']
        if training_args.do_test:
            training_args.train_file = os.environ['SM_CHANNEL_TEST']

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if is_main_process(training_args.local_rank)
        else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.train_file is not None:
        # Loading a dataset from local csv files
        # TODO: perhaps have a `load_klydo_dataset` here that accepts a bucket
        # and key and downloads data from s3, then loads it with `load_datastet`.
        datasets = load_dataset("csv", data_files={"train": data_args.train_file})
    # Currently cannot do training and prediction in the same run
    elif data_args.predict_file is not None:
        datasets = load_dataset("csv", data_files={"test": data_args.predict_file})
    else:
        logger.warning("No train or test file set. Exiting script.")
        return None

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir,
    )
    tokenizer = T5Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir,
    )

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
        max_length = data_args.max_seq_length
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
        max_length = None

    def preprocess_function(examples):
        input_encodings = tokenizer.batch_encode_plus(
            examples["source"],
            padding=padding,
            max_length=128,
            return_tensors="np",
            truncation=True,
        ).input_ids
        target_encodings = tokenizer.batch_encode_plus(
            examples["labels"],
            padding=padding,
            max_length=max_length,
            return_tensors="np",
            truncation=True,
        ).input_ids
        encodings = {"input_ids": input_encodings, "labels": target_encodings}

        return encodings

    datasets = datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    if training_args.do_train:
        train_dataset = datasets["train"]
    else:
        train_dataset = None

    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        labels = p.label_ids
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train:

        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        trainer.train(
            model_path=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.save_model()  # Saves the tokenizer too for easy upload

    if training_args.do_predict:
        logger.info("*** Test ***")
        input_ids = tokenizer(
            "flat - contains - four bedrooms", return_tensors="pt"
        ).input_ids
        outputs = model.generate(input_ids)
        return None

        test_dataset = datasets["test"]
        # Removing the `label` columns because it's not needed for prediction
        try:
            test_dataset.remove_columns_("labels")
        except ValueError:
            pass
        # TODO: Should we use generate here?
        predictions = trainer.predict(test_dataset=test_dataset).predictions

        output_predict_file = os.path.join(
            training_args.output_dir, f"test_results.txt"
        )
        if trainer.is_world_process_zero():
            # TODO: write function for saving and uploading results to s3
            with open(output_predict_file, "w") as writer:
                logger.info(f"***** Test results *****")
                print(predictions)
                # for key in sorted(predictions.keys()):
                #     logger.info("  %s = %s", key, str(predictions[key]))
                #     writer.write("%s = %s\n" % (key, str(predictions[key])))
        return predictions


if __name__ == "__main__":
    main()


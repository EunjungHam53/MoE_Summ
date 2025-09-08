from pathlib import Path
#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning the library models for sequence to sequence.
"""
from args import ModelArguments, DataTrainingArguments, my_Seq2SeqTrainingArguments
from compute_metric import MetricCompute
from transformers.models.bart.tokenization_bart import BartTokenizer
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import (
    HfArgumentParser,
    default_data_collator,
    set_seed
)
from filelock import FileLock
import transformers
from datasets import DatasetDict
import nltk  # Here to have a nice missing dependency error message early on
from transformers import BartForCausalLM
from magic_bart import MyBart, MyDataCollatorForSeq2Seq, MySeq2SeqTrainer
import os
import logging
import pdb
import sys
import traceback
import torch
# import comet
from dataset_maker import DatasetMaker
import glob
import zipfile
from transformers.models.bart.modeling_bart import (
    shift_tokens_right,
    BartConfig,
    BartPretrainedModel,
    BartClassificationHead,
    BartLearnedPositionalEmbedding, BartAttention,
)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

from transformers import GenerationConfig

# from magic_bart import MyBart, MyCometCallback, AutoDecodeCallback, MyDataCollatorForSeq2Seq, MySeq2SeqTrainer,MyBartConfig

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int = None):
    """
    Expands attention_mask from [batch_size, seq_len] to [batch_size, 1, tgt_len, seq_len]
    for multi-head attention use.
    
    Args:
        mask: [batch_size, src_len] - attention mask where 1 = attend, 0 = ignore
        dtype: target data type
        tgt_len: target sequence length (defaults to src_len if None)
    
    Returns:
        expanded_mask: [batch_size, 1, tgt_len, src_len] 
                      with -inf for positions to ignore, 0.0 for positions to attend
    """
    batch_size, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    
    # Expand dims: [batch_size, src_len] -> [batch_size, 1, 1, src_len]
    expanded_mask = mask[:, None, None, :].to(dtype=dtype)
    
    # Expand to target shape: [batch_size, 1, tgt_len, src_len]
    expanded_mask = expanded_mask.expand(batch_size, 1, tgt_len, src_len)
    
    # Invert mask: 1 becomes 0 (attend), 0 becomes 1 (ignore)
    inverted_mask = 1.0 - expanded_mask
    
    # Replace 1s (ignore positions) with -inf, keep 0s (attend positions) as 0
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def _make_causal_mask(input_ids_shape, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Creates a causal (lower triangular) attention mask.
    
    Args:
        input_ids_shape: (batch_size, seq_len)
        dtype: target data type
        past_key_values_length: length of cached keys/values
    
    Returns:
        causal_mask: [batch_size, 1, seq_len, seq_len + past_key_values_length]
                    with -inf for future positions, 0.0 for valid positions
    """
    batch_size, tgt_len = input_ids_shape
    
    # Create causal mask matrix - không cần device parameter vì sẽ .to(device) sau
    mask = torch.full((tgt_len, tgt_len), float('-inf'), dtype=dtype)
    
    # Fill lower triangle (including diagonal) với 0s
    mask = torch.triu(mask, diagonal=1)  # Upper triangular với -inf, lower triangle với 0
    
    # Handle past key values length
    if past_key_values_length > 0:
        # Thêm zeros cho past keys/values
        past_mask = torch.zeros(tgt_len, past_key_values_length, dtype=dtype)
        mask = torch.cat([past_mask, mask], dim=-1)
    
    # Expand to batch dimension: [batch_size, 1, tgt_len, tgt_len + past_key_values_length]
    return mask[None, None, :, :].expand(batch_size, 1, tgt_len, tgt_len + past_key_values_length)
    
logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, my_Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses() 
    training_args.logging_steps = 10
    data_args.log_root = os.path.join(data_args.log_root, data_args.proj_name, data_args.exp_name)

    training_args.output_dir = os.path.join(data_args.log_root, 'model')
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    if training_args.do_train:
        python_list = glob.glob('./*.py')
        zip_file = zipfile.ZipFile(data_args.log_root + '/code.zip', 'w')
        for d in python_list:
            zip_file.write(d)
        for d in glob.glob('dataset/*.py'):
            zip_file.write(d)
        for d in glob.glob('cmd/*.py'):
            zip_file.write(d)
        for d in glob.glob('metrics/*.py'):
            zip_file.write(d)
        for d in glob.glob('src/transformers/moebert/*.py'):
            zip_file.write(d)
        zip_file.close()
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    # logger.info("Training/evaluation parameters %s", training_args)
    # logger.info("Dataset parameters %s", data_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if not training_args.do_train and (
            training_args.do_eval or training_args.do_predict) and model_args.model_name_or_path is None:
        # 纯测试且没指定ckpt 就用最新的ckpt
        model_args.model_name_or_path = last_checkpoint if last_checkpoint is not None else get_last_checkpoint(
            training_args.output_dir)
    if training_args.do_train and last_checkpoint is not None:
        logger.warning(f'using previous checkpoint {last_checkpoint}')
        model_args.model_name_or_path = last_checkpoint

    logger.info(f'******* Loading model form pretrained {model_args.model_name_or_path} **********')
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)  # 如果用bart-base就用这行
    logger.info('load BartTokenizer')

    config = BartConfig.from_pretrained(model_args.model_name_or_path)
    config.vocab_size = 64001  # Thêm dòng này để khớp với checkpoint
    
    config.intermediate_size = model_args.intermediate_size
    config.route_method = model_args.route_method
    config.num_experts = model_args.num_experts
    config.num_datasets=model_args.num_datasets
    config.margin_loss=model_args.margin_loss
    config.moe_model = model_args.moe_model
    config.moe_model_enc = model_args.moe_model_enc
    config.moe_load = training_args.moe_load
    config.share_importance = model_args.share_importance
    config.keep_resident = model_args.keep_resident

    # Đảm bảo vocab_size đúng (BARTPho-word-base = 50265)
    config.vocab_size = tokenizer.vocab_size

    # Thêm dòng này cho MoEBERT
    config.moebert_load_experts = training_args.moe_load
    config.moebert = model_args.moe_model
    
    training_args.margin_loss = model_args.margin_loss
    
    if hasattr(config, 'vocab_size'):
        config.vocab_size = 64001
    model = MyBart.from_pretrained(model_args.model_name_or_path,config=config)

    logger.info('load model')

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if data_args.save_dataset_path is None and data_args.gene_dataset_path:
        maker = DatasetMaker(data_args.gene_dataset_path, data_args, training_args, tokenizer)
        datasets = maker.make_dataset()
    else:
        logger.info(f'******* Loading Dataset from {data_args.save_dataset_path} **********')
        datasets = DatasetDict.load_from_disk(data_args.save_dataset_path)

    train_dataset = datasets["train"] if training_args.do_train is not None and "train" in datasets else None
    eval_dataset = datasets["test"] if training_args.do_eval is not None and "validation" in datasets else None
    test_dataset = datasets["test"] if training_args.do_predict is not None and "test" in datasets else datasets[
        "validation"]
    if training_args.do_predict is None and "test" not in datasets:
        logging.warning(f'using validation dataset as test!')

    if data_args.max_val_samples is not None:
        test_dataset = test_dataset.select(range(data_args.max_val_samples))

    max_target_length = data_args.val_max_target_length
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = MyDataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    comp_metric = MetricCompute(data_args, tokenizer, test_dataset, eval_dataset)

    # comet_callback = MyCometCallback(data_args.proj_name, data_args.exp_name)

    # model.config.num_beams = data_args.num_beams
    # model.config.max_length = data_args.max_target_length

    

    # Thay vì set trong model config
    generation_config = GenerationConfig(
        max_length=data_args.max_target_length,
        num_beams=data_args.num_beams,
        decoder_start_token_id=tokenizer.bos_token_id or 0,
        pad_token_id=tokenizer.pad_token_id or 1,
        eos_token_id=tokenizer.eos_token_id or 2
    )
    model.generation_config = generation_config
        

    # for param in model.bart.parameters():
    #     param.requires_grad = False

    # for arg_class in [model_args, data_args, training_args, model.config]:
    #     for k, v in arg_class.to_dict().items():
    #         comet_callback.exp.experiment.log_parameter(k, v)
    # python_list = glob.glob('./*.py')
    # for file in python_list:
    #     comet_callback.exp.experiment.log_code(file_name=file, folder='./', code=None, code_name=None)

    # Initialize our Trainer
    # pdb.set_trace()
    # if training_args.predict_with_generate:
    #     training_args.report_to = ['comet_ml']
    if training_args.freeze:
        for name, param in model.model.named_parameters():
            if 'gate_weight' in name or 'expert' in name:
                param.requires_grad = True
                print (str(name))
            else:
                param.requires_grad = False
            # if 'gate_weight' in name or 'expert' in name or 'layer_norm' in name or 'out_proj' in name\
            #        :
            #     param.requires_grad = True
            #     print (str(name))
            # else:
            #     param.requires_grad = False

            # if ('fc1' in name or 'fc2' in name) and 'expert' not in name:
            #     param.requires_grad = False
            #     print(str(name))
            # else:
            #     param.requires_grad = True


    trainer = MySeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=comp_metric.compute_metrics if training_args.predict_with_generate else None,
        # callbacks=[comet_callback]  # auto_decode_callback
    )
    comp_metric.trainer = trainer
    # comet_callback.set_trainer(trainer)

    # Training
    if training_args.do_train:
        try:
            if last_checkpoint is not None:  # 如果是继续之前的训练需要加载步数和optimizer
                train_result = trainer.train(
                    resume_from_checkpoint=model_args.model_name_or_path)  # resume_from_checkpoint=checkpoint
            else:
                train_result = trainer.train()
            logger.info("***** Train results *****")
            # for key, value in sorted(train_result.metrics.items()):
            #     logger.info(f"  {key} = {value}")
        except KeyboardInterrupt:
            logger.info('stop training')
        finally:
            traceback.print_exc()
            if trainer.is_world_process_zero():
                logger.info('exit, saving model')
                # pdb.set_trace()
                trainer.save_model(output_dir=os.path.join(training_args.output_dir, f'checkpoint-{trainer.state.global_step}'))  # Saves the tokenizer too for easy upload
                # tokenizer.save_pretrained(os.path.join(training_args.output_dir, f'checkpoint-{trainer.state.global_step}'))
                trainer.state.save_to_json(
                    os.path.join(training_args.output_dir, f'checkpoint-{trainer.state.global_step}',
                                 'trainer_state.json'))
            exit(0)

    # predict
    if training_args.do_predict:
        logger.info(f"*** Test ***")
        trainer.state.global_step = model_args.model_name_or_path.split('-')[-1]
        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        print(test_results.metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                test_results.label_ids[test_results.label_ids < 0] = tokenizer.pad_token_id
                test_label = tokenizer.batch_decode(
                    test_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                test_preds = tokenizer.batch_decode(
                    test_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                test_preds = [pred.strip() for pred in test_preds]
                test_labels = [label.strip() for label in test_label]
                
                # In ra 10 ví dụ đầu để debug
                for pred, lab in zip(test_preds[:10], test_labels[:10]):
                    logger.info(f'PRED: {pred}')
                    logger.info(f'GOLD: {lab}')
                    logger.info('---')
        
                dec_dir = Path(data_args.log_root) / f'decode-{trainer.state.global_step}'
                dec_dir.mkdir(parents=True, exist_ok=True)
                
                ref_file = dec_dir / 'reference.txt'
                dec_file = dec_dir / 'decoded.txt'
                
                with ref_file.open('w', encoding='utf8') as fo_ref, \
                     dec_file.open('w', encoding='utf8') as fo_dec:
                    for pred, lab in zip(test_preds, test_labels):
                        print(lab, file=fo_ref)
                        print(pred, file=fo_dec)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    with open('mybart.pid', 'w', encoding='utf8') as w:
        w.write(str(os.getpid()))
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main()
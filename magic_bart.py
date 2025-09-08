import torch
import collections
import logging
import math
import os
import pdb
import random
import subprocess
import sys
from typing import Optional, Tuple, Union, Dict, Any, List

from attr import dataclass
from transformers import BartTokenizer

from transformers import PreTrainedTokenizerBase
from transformers.file_utils import ModelOutput
from transformers.tokenization_utils_base import PaddingStrategy
from transformers import GenerationMixin

# from comet import MyExperiment
import torch
from torch.nn import CrossEntropyLoss
from torch import nn
import torch.nn.functional as F
from packaging import version
from transformers import BartForConditionalGeneration

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from transformers.trainer import Trainer
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer_callback import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, \
    ProgressCallback
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput, \
    BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bart.modeling_bart import (
    shift_tokens_right,
    BartConfig,
    BartPretrainedModel,
    BartClassificationHead,
    BartLearnedPositionalEmbedding, BartAttention,
)
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.moebert.utils import (
    ImportanceProcessor,
    MoEModelOutput,
    MoEModelOutputWithPooling,
)

# from transformers.moebert.moe_layer import MoELayer

logger = logging.getLogger(__name__)

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

class MyBartEncoder(BartPretrainedModel):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([MyBartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        self.post_init()

    def post_init(self):
        """Initialize weights and apply final processing."""
        # Cho pretrained model, chỉ cần pass hoặc gọi parent
        super().post_init() if hasattr(super(), 'post_init') else None

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            idxes=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        positions = torch.arange(0, input_shape[-1], dtype=torch.long, device=inputs_embeds.device)
        embed_pos = self.embed_positions(positions.unsqueeze(0).expand(input_shape[0], -1))

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class MyMoeEncoder(BartPretrainedModel):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        # self.layers = nn.ModuleList([MyBartEncoderLayer(config) for _ in range(config.encoder_layers)])

        self.layers = nn.ModuleList([MyMoeEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        self.post_init()

    def post_init(self):
        """Initialize weights and apply final processing."""
        # Cho pretrained model, chỉ cần pass hoặc gọi parent
        super().post_init() if hasattr(super(), 'post_init') else None

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            idxes=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        positions = torch.arange(0, input_shape[-1], dtype=torch.long, device=inputs_embeds.device)
        embed_pos = self.embed_positions(positions.unsqueeze(0).expand(input_shape[0], -1))

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                        idxes=idxes
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class MyBartEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            layer_head_mask: torch.Tensor,
            output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class MyMoeEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.config=config
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.experts_fc1 = nn.ModuleList(
            [nn.Linear(config.d_model, config.intermediate_size) for _ in range(config.num_experts)])
        self.experts_fc2 = nn.ModuleList(
            [nn.Linear(config.intermediate_size, config.d_model) for _ in range(config.num_experts)])
        self.gate_weight = nn.ModuleList(
            [nn.Linear(config.d_model, config.num_experts) for _ in range(config.num_datasets)])

        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            layer_head_mask: torch.Tensor,
            output_attentions: bool = False,
            idxes=None
    ):
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        if self.config.margin_loss:
            # Fully Connected
            residual = hidden_states
            ori_hidden_states = self.activation_fn(self.fc1(hidden_states))
            ori_hidden_states = nn.functional.dropout(ori_hidden_states, p=self.activation_dropout,
                                                      training=self.training)
            ori_hidden_states = self.fc2(ori_hidden_states)
            ori_hidden_states = nn.functional.dropout(ori_hidden_states, p=self.dropout, training=self.training)
            ori_hidden_states = residual + ori_hidden_states
            ori_hidden_states = self.final_layer_norm(ori_hidden_states)
        else:
            ori_hidden_states = None

        hidden_states, gate = self.feed_forward(hidden_states, idxes)

        if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    def feed_forward(self, x, idxes):
        expert_list = dict()
        if not self.config.moe_load:
            for i in range(self.config.num_experts):
                expert_list[i] = []
                expert_list[i].append(torch.cat([self.fc1.weight, self.experts_fc1[i].weight], 0))  # fc1 weight
                expert_list[i].append(torch.cat([self.fc1.bias, self.experts_fc1[i].bias], 0))
                expert_list[i].append(torch.cat([self.fc2.weight, self.experts_fc2[i].weight], 1))
    
        bsz, seq_len, dim = x.size()
        original_shape = (bsz, seq_len, dim)
        x = x.view(-1, dim)
        total_tokens = x.size(0)  # Should be bsz * seq_len
    
        # Fix: Handle idxes dimension mismatch
        if idxes.size(0) != bsz:
            # During beam search or other scenarios, batch size may differ
            if hasattr(self.config, 'num_beams') and self.config.num_beams > 1:
                # Assume idxes corresponds to original batch size
                original_batch_size = idxes.size(0)
                num_beams = bsz // original_batch_size
                idxes = idxes.unsqueeze(-1).repeat(1, num_beams).view(-1)
            else:
                # Fallback: repeat or truncate idxes to match bsz
                if idxes.size(0) < bsz:
                    repeat_times = (bsz + idxes.size(0) - 1) // idxes.size(0)  # Ceiling division
                    idxes = idxes.repeat(repeat_times)[:bsz]
                else:
                    idxes = idxes[:bsz]
        
        # Expand idxes to match total tokens
        idxes = idxes.unsqueeze(-1).repeat(1, seq_len).view(-1)
        
        # Verify dimensions match
        assert idxes.size(0) == total_tokens, f"idxes size {idxes.size(0)} != total_tokens {total_tokens}"
        
        order = idxes.argsort(0)
    
        num_tokens = F.one_hot(idxes, self.config.num_datasets).gt(0).sum(0)
        x_for_gate = x[order]
        x_for_gate = x_for_gate.split(num_tokens.tolist(), dim=0)
        gate = torch.empty(0, device=x.device, dtype=torch.long)
        gate_value = torch.empty(0, device=x.device)
    
        for i in range(self.config.num_datasets):
            if x_for_gate[i].size(0) != 0:
                hidden_states = self.gate_weight[i](x_for_gate[i])
                hidden_states = F.softmax(hidden_states, -1)  # case_num,expert_num
                select_gate_value = torch.topk(hidden_states, 1, 1)[0].squeeze(-1)
                select_gate_index = torch.topk(hidden_states, 1, 1)[1].squeeze(-1)
                gate = torch.cat([gate, select_gate_index])
                gate_value = torch.cat([gate_value, select_gate_value])
        gate = gate[order.argsort(0)].to(torch.int64)
        gate_value = gate_value[order.argsort(0)]
    
        ##################
        order = gate.argsort(0)
        num_tokens = F.one_hot(gate, self.config.num_experts).gt(0).sum(0)
        x = x[order]  # reorder according to expert number
        gate_value = gate_value[order]
        x = x.split(num_tokens.tolist(), dim=0)  # a list of length self.num_experts
        gate_value = gate_value.split(num_tokens.tolist(), dim=0)
        new_x = []
        for i in range(self.config.num_experts):
            if x[i].size(0) == 0:  # Skip empty expert
                continue
                
            residual = x[i]  # batch, d_model
            hidden_states = self.activation_fn(
                F.linear(x[i], expert_list[i][0], expert_list[i][1]))  # batch,intermediate
            hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
            hidden_states = F.linear(hidden_states, expert_list[i][2], self.fc2.bias)
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = hidden_states * (gate_value[i].unsqueeze(-1))
            new_x.append(hidden_states)
    
        x = torch.vstack(new_x)
        x = x[order.argsort(0)]  # restore original order
        
        # Ensure we have the right number of elements before reshaping
        assert x.size(0) == total_tokens, f"Final tensor size {x.size(0)} != expected {total_tokens}"
        x = x.view(original_shape)
        
        return x, gate


class MyBartDecoder(BartPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([MyBartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=None,
            encoder_head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            idxes=None
            # addi_source_encoder_hidden_states=None,
            # addi_source_encoder_attention_mask=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            encoder_head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # expand encoder attention mask
        # if addi_source_encoder_hidden_states is not None and addi_source_encoder_attention_mask is not None:
        #     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        #     addi_source_encoder_attention_mask = _expand_mask(addi_source_encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        # positions = self.embed_positions(input_shape, past_key_values_length)

        seq_length = input_shape[-1]
        position_ids = torch.arange(
            past_key_values_length, 
            seq_length + past_key_values_length, 
            dtype=torch.long, 
            device=inputs_embeds.device
        ).unsqueeze(0).expand(input_shape[0], -1)
        
        positions = self.embed_positions(position_ids)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    encoder_head_mask[idx] if encoder_head_mask is not None else None,
                    None,
                    # addi_source_encoder_hidden_states,
                    # addi_source_encoder_attention_mask,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    # addi_source_encoder_hidden_states=addi_source_encoder_hidden_states,
                    # addi_source_encoder_attention_mask=addi_source_encoder_attention_mask,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return MoEModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class MyMoeDecoder(BartPretrainedModel):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([MyMoeDecoderLayer(config, i) for i in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=None,
            encoder_head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            idxes=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # expand encoder attention mask
        # if addi_source_encoder_hidden_states is not None and addi_source_encoder_attention_mask is not None:
        #     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        #     addi_source_encoder_attention_mask = _expand_mask(addi_source_encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        # positions = self.embed_positions(input_shape, past_key_values_length)

        seq_length = input_shape[-1]
        position_ids = torch.arange(
            past_key_values_length, 
            seq_length + past_key_values_length, 
            dtype=torch.long, 
            device=inputs_embeds.device
        ).unsqueeze(0).expand(input_shape[0], -1)
        
        positions = self.embed_positions(position_ids)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    encoder_head_mask[idx] if encoder_head_mask is not None else None,
                    None,
                    # addi_source_encoder_hidden_states,
                    # addi_source_encoder_attention_mask,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    idxes=idxes
                )
            # hidden_states = layer_outputs[0]
            # ori_hidden_states = layer_outputs[1]

            # if use_cache:
            #     # next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)
            #     next_decoder_cache += (layer_outputs[3 if output_attentions else 2],)

            hidden_states = layer_outputs[0]
            if len(layer_outputs) > 1:
                ori_hidden_states = layer_outputs[1]
            else:
                ori_hidden_states = None
            
            # Xử lý cache index đúng cách
            if use_cache:
                cache_index = len(layer_outputs) - 1  # present_key_value luôn là element cuối
                next_decoder_cache += (layer_outputs[cache_index],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )

        return MoEModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
            ori_hidden_states=ori_hidden_states
        )


class MyBartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )

        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            encoder_layer_head_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
            # addi_source_encoder_hidden_states=None,
            # addi_source_encoder_attention_mask=None,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            encoder_layer_head_mask (:obj:`torch.FloatTensor`): mask for encoder attention heads in a given layer of
                size `(config.encoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MyMoeDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig, layer_idx=-100):
        super().__init__()
        self.embed_dim = config.d_model
        self.config = config
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )

        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        # ffn = FeedForward(config)
        self.experts_fc1 = nn.ModuleList(
            [nn.Linear(config.d_model, config.intermediate_size) for _ in range(config.num_experts)])
        self.experts_fc2 = nn.ModuleList(
            [nn.Linear(config.intermediate_size, config.d_model) for _ in range(config.num_experts)])
        self.gate_weight = nn.ModuleList(
            [nn.Linear(config.d_model, config.num_experts) for _ in range(config.num_datasets)])

        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            encoder_layer_head_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
            idxes=None
    ):
        residual = hidden_states
        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            # FIX: Check if present_key_value and cross_attn_present_key_value are not None
            if present_key_value is not None and cross_attn_present_key_value is not None:
                present_key_value = present_key_value + cross_attn_present_key_value
            elif cross_attn_present_key_value is not None:
                present_key_value = cross_attn_present_key_value

        if self.config.margin_loss:
            # Fully Connected
            residual = hidden_states
            ori_hidden_states = self.activation_fn(self.fc1(hidden_states))
            ori_hidden_states = F.dropout(ori_hidden_states, p=self.activation_dropout, training=self.training)
            ori_hidden_states = self.fc2(ori_hidden_states)
            ori_hidden_states = F.dropout(ori_hidden_states, p=self.dropout, training=self.training)
            ori_hidden_states = residual + ori_hidden_states
            ori_hidden_states = self.final_layer_norm(ori_hidden_states)
        else:
            ori_hidden_states = None

        hidden_states, gate = self.feed_forward(hidden_states, idxes)

        outputs = (hidden_states, ori_hidden_states)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def feed_forward(self, x, idxes):
        expert_list = dict()
        if not self.config.moe_load:
            for i in range(self.config.num_experts):
                expert_list[i] = []
                expert_list[i].append(torch.cat([self.fc1.weight, self.experts_fc1[i].weight], 0))  # fc1 weight
                expert_list[i].append(torch.cat([self.fc1.bias, self.experts_fc1[i].bias], 0))
                expert_list[i].append(torch.cat([self.fc2.weight, self.experts_fc2[i].weight], 1))
    
        bsz, seq_len, dim = x.size()
        original_shape = (bsz, seq_len, dim)
        x = x.view(-1, dim)
        total_tokens = x.size(0)  # Should be bsz * seq_len
    
        # Fix: Handle idxes dimension mismatch during beam search
        if idxes.size(0) != bsz:
            # During beam search, batch size is multiplied by num_beams
            if hasattr(self.config, 'num_beams') and self.config.num_beams > 1:
                # Assume idxes corresponds to original batch size
                original_batch_size = idxes.size(0)
                num_beams = bsz // original_batch_size
                idxes = idxes.unsqueeze(-1).repeat(1, num_beams).view(-1)
            else:
                # Fallback: repeat or truncate idxes to match bsz
                if idxes.size(0) < bsz:
                    repeat_times = (bsz + idxes.size(0) - 1) // idxes.size(0)  # Ceiling division
                    idxes = idxes.repeat(repeat_times)[:bsz]
                else:
                    idxes = idxes[:bsz]
        
        # Expand idxes to match total tokens
        idxes = idxes.unsqueeze(-1).repeat(1, seq_len).view(-1)
        
        # Verify dimensions match
        assert idxes.size(0) == total_tokens, f"idxes size {idxes.size(0)} != total_tokens {total_tokens}"
        
        order = idxes.argsort(0)
    
        num_tokens = F.one_hot(idxes, self.config.num_datasets).gt(0).sum(0)
        x_for_gate = x[order]
        x_for_gate = x_for_gate.split(num_tokens.tolist(), dim=0)
        gate = torch.empty(0, device=x.device, dtype=torch.long)
        gate_value = torch.empty(0, device=x.device)
    
        for i in range(self.config.num_datasets):
            if x_for_gate[i].size(0) != 0:
                hidden_states = self.gate_weight[i](x_for_gate[i])
                hidden_states = F.softmax(hidden_states, -1)  # case_num,expert_num
                select_gate_value = torch.topk(hidden_states, 1, 1)[0].squeeze(-1)
                select_gate_index = torch.topk(hidden_states, 1, 1)[1].squeeze(-1)
                gate = torch.cat([gate, select_gate_index])
                gate_value = torch.cat([gate_value, select_gate_value])
        gate = gate[order.argsort(0)].to(torch.int64)
        gate_value = gate_value[order.argsort(0)]
    
        ##################
        order = gate.argsort(0)
        num_tokens = F.one_hot(gate, self.config.num_experts).gt(0).sum(0)
        x = x[order]  # reorder according to expert number
        gate_value = gate_value[order]
        x = x.split(num_tokens.tolist(), dim=0)  # a list of length self.num_experts
        gate_value = gate_value.split(num_tokens.tolist(), dim=0)
        new_x = []
        for i in range(self.config.num_experts):
            if x[i].size(0) == 0:  # Skip empty expert
                continue
                
            residual = x[i]  # batch, d_model
            hidden_states = self.activation_fn(
                F.linear(x[i], expert_list[i][0], expert_list[i][1]))  # batch,intermediate
            hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
            hidden_states = F.linear(hidden_states, expert_list[i][2], self.fc2.bias)
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            
            if self.config.keep_resident:
                hidden_states = hidden_states * (gate_value[i].unsqueeze(-1))
                hidden_states = residual + hidden_states
                hidden_states = self.final_layer_norm(hidden_states)
            else:
                hidden_states = residual + hidden_states
                hidden_states = self.final_layer_norm(hidden_states)
                hidden_states = hidden_states * (gate_value[i].unsqueeze(-1))
            new_x.append(hidden_states)
    
        x = torch.vstack(new_x)
        x = x[order.argsort(0)]  # restore original order
        
        # Ensure we have the right number of elements before reshaping
        assert x.size(0) == total_tokens, f"Final tensor size {x.size(0)} != expected {total_tokens}"
        x = x.view(original_shape)
        
        return x, gate


class MaskBartAttention(BartAttention):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attention_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_float = torch.sigmoid(attn_weights)
        attn_weights = attn_weights_float.type_as(attn_weights)
        return attn_weights


class MyBartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        if config.moe_model_enc:
            self.encoder = MyMoeEncoder(config, self.shared)
        else:
            self.encoder = MyBartEncoder(config, self.shared)

        if config.moe_model:
            self.decoder = MyMoeDecoder(config, self.shared)
        else:
            self.decoder = MyBartDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            idxes=None
            # addi_source=None,
            # addi_source_attention_mask=None,
            # addi_source_encoder_outputs=None,
    ):
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                idxes=idxes
            )
            # addi_source_encoder_outputs = self.encoder(
            #     input_ids=addi_source,
            #     attention_mask=addi_source_attention_mask
            # )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
            # addi_source_encoder_outputs = BaseModelOutput(
            #     last_hidden_state=addi_source_encoder_outputs[0],
            #     hidden_states=addi_source_encoder_outputs[1] if len(addi_source_encoder_outputs) > 1 else None,
            #     attentions=addi_source_encoder_outputs[2] if len(addi_source_encoder_outputs) > 2 else None,
            # )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            idxes=idxes
            # addi_source_encoder_hidden_states=addi_source_encoder_outputs[0],
            # addi_source_encoder_attention_mask=addi_source_attention_mask,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            ori_hidden_states=decoder_outputs.ori_hidden_states
        )


class MyBart(BartPretrainedModel, GenerationMixin):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    _tied_weights_keys = [
        "model.encoder.embed_tokens.weight",
        "model.decoder.embed_tokens.weight", 
        "model.shared.weight"
    ]

    # THÊM DÒNG NÀY ĐỂ FIX LỖI
    _keys_to_ignore_on_save = [r"lm_head.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)

        self.model = MyBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # THÊM DÒNG NÀY để unshare weights

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            seg_labels=None,
            sent_indexs=None,
            idxes=None
            # addi_source=None,
            # addi_source_attention_mask=None,
            # addi_source_encoder_outputs=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            idxes=idxes
            # addi_source=addi_source,
            # addi_source_attention_mask=addi_source_attention_mask,
        )  # dict (['last_hidden_state', 'past_key_values', 'encoder_last_hidden_state']
        # [0]:[8, 99, 1024]
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias  # [1,vocab_size]
        if outputs.ori_hidden_states is not None:
            zero_logits = self.lm_head(outputs.ori_hidden_states) + self.final_logits_bias  # [1,vocab_size]
        else:
            zero_logits = None
        masked_lm_loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction='mean')
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            zero_logits=zero_logits
        )

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, 
        inputs_tensor: torch.Tensor, 
        model_kwargs, 
        model_input_name: Optional[str] = None,
        generation_config = None
    ) -> Dict[str, Any]:
        # retrieve encoder hidden states
        encoder = self.get_encoder()
        # Prepare encoder kwargs by removing decoder-specific arguments
        encoder_kwargs = {
            argument: value for argument, value in model_kwargs.items() 
            if not argument.startswith("decoder_") 
            and argument not in ["use_cache", "output_attentions", "output_hidden_states"]
            and not argument.startswith("cross_attn")
        }
        
        # Handle the input tensor - could be input_ids or inputs_embeds
        if model_input_name is None:
            model_input_name = self.main_input_name
        
        encoder_kwargs[model_input_name] = inputs_tensor
        
        # Add standard generation arguments if they exist in model_kwargs
        for key in ["attention_mask", "head_mask", "inputs_embeds", "output_attentions", "output_hidden_states"]:
            if key in model_kwargs:
                encoder_kwargs[key] = model_kwargs[key]
        
        # Add custom arguments specific to your model
        if "idxes" in model_kwargs:
            encoder_kwargs["idxes"] = model_kwargs["idxes"]
        
        model_kwargs["encoder_outputs"] = encoder(return_dict=True, **encoder_kwargs)
        
        return model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
            input_ids: torch.LongTensor,
            expand_size: int = 1,
            is_encoder_decoder: bool = False,
            attention_mask: torch.LongTensor = None,
            encoder_outputs: ModelOutput = None,
            # addi_source_encoder_outputs: ModelOutput = None,
            **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

            # addi_source_encoder_outputs["last_hidden_state"] = addi_source_encoder_outputs.last_hidden_state.index_select(
            #     0, expanded_return_idx.to(addi_source_encoder_outputs.last_hidden_state.device)
            # )
            # model_kwargs['addi_source_encoder_outputs'] = addi_source_encoder_outputs
        return input_ids, model_kwargs

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "idxes": kwargs['idxes'],
        }

    @staticmethod
    def _force_token_id_to_be_generated(scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(scores.shape[1]) if x != token_id]] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # Fix: Check if layer_past is None
            if layer_past is None:
                reordered_past += (None,)
                continue
            
            # cached cross_attention states don't have to be reordered -> they are always the same
            # Fix: Ensure we have enough elements before slicing
            if len(layer_past) >= 3:
                reordered_past += (
                    tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:3]) + layer_past[3:],
                )
            else:
                # Handle case where layer_past has fewer than 3 elements
                reordered_past += (
                    tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
                )
        return reordered_past

@dataclass
class MyDataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    model_args: object = None
    hash_list = dict()
    # Vietnamese hash list
    hash_list['Văn hóa - Xã hội'] = 0
    hash_list['Pháp luật'] = 1
    hash_list['Kinh tế'] = 2
    hash_list['Khoa học - Công nghệ'] = 3
    hash_list['Giải trí - Thể thao'] = 4
    hash_list['Đời sống'] = 5
    hash_list['Thế giới'] = 6
    hash_list['Giáo dục'] = 7

    def __call__(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        idxes = [f['idxes'] for f in features]
        for f in features:
            for k in ['adomain', 'qdomain', 'summary', 'token_type_ids', 'retrieval', 'content', 'selftext',
                      'subreddit',
                      'answers', 'title', 'idxes']:
                if k in f:
                    del f[k]

        to_return = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        idxes = [self.hash_list[idx] for idx in idxes]
        idxes = torch.tensor(idxes)
        idxes = idxes.to(torch.int64)
        to_return['idxes'] = idxes
        return to_return


class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels, weight=None):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
        if weight is not None:
            nll_loss = nll_loss * weight
            smoothed_loss = smoothed_loss * weight

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


class MySeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        if 'tokenizer' in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')
        super().__init__(*args, **kwargs)
        self.label_smoother = LabelSmoother()
        self.importance_ffn = None
        self.importance_ffn_enc = None

    def compute_margin_loss(self, zero_logits, new_logits, labels):
        if labels.dim() == zero_logits.dim() - 1:
            labels = labels.unsqueeze(-1)
        padding_mask = labels.eq(-100)
        labels = torch.clamp(labels, min=0)

        zero_logits = nn.functional.softmax(zero_logits, dim=-1)
        zero_logits = zero_logits.gather(dim=-1, index=labels)
        zero_logits.masked_fill_(padding_mask, 0.0)  # [4, 84, 1]
        lm_preds = zero_logits.squeeze(2).contiguous()  # batch_size, len

        new_logits = nn.functional.softmax(new_logits, dim=-1)
        new_logits = new_logits.gather(dim=-1, index=labels)
        new_logits.masked_fill_(padding_mask, 0.0)  # [4, 84, 1]
        new_preds = new_logits.squeeze(2).contiguous()  # batch_size, len
        delta = new_preds - lm_preds

        new_lm = (1 - new_preds).mul(1 - (new_preds - lm_preds) ** 5) / 2  # [4, 84]
        padding_mask = padding_mask.squeeze(-1)
        new_lm.masked_fill_(padding_mask, 0.0)
        new_lm = new_lm.sum()
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        new_lm = new_lm / (num_active_elements)

        return delta, new_lm

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs["labels"]
        else:
            labels = None
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        if self.args.weight:
            idxes = inputs['idxes']  # batch
            bincount = torch.bincount(idxes)
            bincount = idxes.size()[0] - bincount
            weight = F.softmax(bincount / self.args.temperature)  # num_expert 3
            weight = weight[idxes].unsqueeze(-1)  # batch, 1
            weight = weight.repeat([1, labels.size()[1]])  # batch,tgt_len
            weight = weight.unsqueeze(-1)
        else:
            weight = None

        if labels is not None:
            loss = self.label_smoother(outputs, labels, weight)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.margin_loss:
            zero_logits = outputs['zero_logits']
            new_logits = outputs['logits']
            delta, new_lm = self.compute_margin_loss(zero_logits, new_logits, labels)
            loss += new_lm * 10
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.
    
        Subclass and override to inject custom behavior.
    
        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
    
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
    
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
    
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        
        # Fix: Use generation_config instead of non-existent attributes
        if hasattr(self.model, 'generation_config') and self.model.generation_config is not None:
            max_length = self.model.generation_config.max_length
            num_beams = self.model.generation_config.num_beams
        else:
            # Fallback values
            max_length = getattr(self.model.config, 'max_length', 512)
            num_beams = getattr(self.model.config, 'num_beams', 4)
        
        gen_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "repetition_penalty": 2.0,
            "no_repeat_ngram_size": 3,
            "idxes": inputs['idxes'],
            "length_penalty": 0.8,
            # Fix: Add required tokens for generation
            "decoder_start_token_id": getattr(self.model.config, 'decoder_start_token_id', 
                                            getattr(self.tokenizer, 'bos_token_id', 0)),
            "pad_token_id": getattr(self.model.config, 'pad_token_id', 
                                  getattr(self.tokenizer, 'pad_token_id', 1)),
            "eos_token_id": getattr(self.model.config, 'eos_token_id', 
                                  getattr(self.tokenizer, 'eos_token_id', 2))
            # "addi_source_attention_mask": inputs['addi_source_attention_mask'],
        }
        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        # Xem lại khi nào dùng autocast ở đây
        # with torch.no_grad():
        if hasattr(self.args, 'fp16') and self.args.fp16:
            with autocast():
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
        loss = outputs["loss"]
    
        if self.args.prediction_loss_only:
            return (loss, None, None)
    
        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
    
        loss = loss.detach()
        # generated_tokens=None
        return (loss, generated_tokens, labels)


if __name__ == '__main__':
    path = '/home/gaoshen.gao/pretrain/antbart-ckpt-40000'
    tokenizer = BertTokenizerFast.from_pretrained(path)
    # model = BartForConditionalGeneration.from_pretrained(path)
    model = MyBart.from_pretrained(path)

    TXT = f"周三市场呈现开盘指数小幅高开，盘中银行、券商、房地产等权重板块带动拉升" + tokenizer.eos_token

    input_ids = tokenizer([TXT], return_tensors='pt', add_special_tokens=False)['input_ids']
    print('-------call--------')
    logits = model(input_ids).logits  # type: torch.Tensor
    print(logits.shape)
    print('Greedy --> ', tokenizer.decode(logits[0].softmax(dim=1).argmax(dim=1)))
    print('-------generate--------')

    summary_ids = model.generate(input_ids, num_beams=4, max_length=100, early_stopping=True)
    print(tokenizer.decode(summary_ids[0], clean_up_tokenization_spaces=False, skip_special_tokens=True))
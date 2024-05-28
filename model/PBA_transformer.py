from transformers.models.switch_transformers.modeling_switch_transformers import *
from transformers.models.switch_transformers.configuration_switch_transformers import SwitchTransformersConfig
import copy
import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

# Position and Behavior Aware Transformer (PBATransformer)
class PBATransformerConfig(SwitchTransformersConfig):
    def __init__(self, 
                 behavior_injection: bool = False,
                 behavior_embedding_dim: int = 64,
                 num_positions: int = 4,
                 num_behavior: int = 4,
                 n_positions: int = 50,
                 sparse_layers_encoder: list = [],
                 sparse_layers_decoder: list = [],
                 behavior_injection_encoder: list = [],
                 behavior_injection_decoder: list = [],
                 Moe_behavior_only: bool = False,
                 shared_expert: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.behavior_injection = behavior_injection
        self.behavior_embedding_dim = behavior_embedding_dim
        self.num_positions = num_positions
        self.num_experts = num_positions + 1 # 1 for the BOS, EOS, PAD tokens
        self.num_behavior = num_behavior
        self.n_positions = n_positions
        self.sparse_layers_encoder = sparse_layers_encoder
        self.sparse_layers_decoder = sparse_layers_decoder
        self.behavior_injection_encoder = behavior_injection_encoder
        self.behavior_injection_decoder = behavior_injection_decoder
        self.Moe_behavior_only = Moe_behavior_only
        self.shared_expert = shared_expert

class PBAEncoderRouter(nn.Module):
    """
    Router takes in the original input ids and generate the position router index and behavior tokens for each token.
    This is not the same as the Switch Transformers router.

    """
    def __init__(self, num_items, config: PBATransformerConfig):
        super().__init__()
        self.num_items = num_items
        self.num_experts = config.num_experts
        self.num_positions = config.num_positions
        self.num_behavior = config.num_behavior
        self.eos = config.eos_token_id
        self.pad = config.pad_token_id
        if config.Moe_behavior_only:
            self.pre_generated_position_index = torch.cat([torch.zeros(1, dtype=torch.long), (torch.tensor([0, 1, 1, 1], dtype=torch.long) + 1).repeat(self.num_items),torch.zeros(1, dtype=torch.long)])
        else:
            self.pre_generated_position_index = torch.cat([torch.zeros(1, dtype=torch.long), (torch.arange(self.num_positions, dtype=torch.long) + 1).repeat(self.num_items),torch.zeros(1, dtype=torch.long)])
        self.behavior_token_idices = torch.arange(0, self.num_items * self.num_positions, self.num_positions) + 1

    def forward(self, input_id_sequence: torch.Tensor) -> Tuple:
        r"""
        We assume that the input ids takes the form of [user_id, behavior_id, item_id1, item_id2, item_idn, behavior_id, item_id1, ..., EOS, PAD, ...]
        the mapped position index will be [0, 1, 2, 3, 4, 1, 2, ..., 0, 0, ...]
        the mapped behavior index will be [0, 0, behavior_id, behavior_id, behavior_id, 0, behavior_id, ..., 0, 0, ...]
        """
        batch_size, seq_length = input_id_sequence.size()
        position_index = self.pre_generated_position_index.to(input_id_sequence.device).repeat(batch_size, 1)
        position_index[(input_id_sequence == self.pad) | (input_id_sequence == self.eos)] = 0
        behavior_indices = self.behavior_token_idices.to(input_id_sequence.device)
        behavior_tokens = input_id_sequence[:, behavior_indices]
        repeat_behavior_tokens = torch.repeat_interleave(behavior_tokens, self.num_positions, dim=1)
        repeat_behavior_tokens = torch.cat((torch.zeros(batch_size, 1, dtype=torch.long, device = input_id_sequence.device), repeat_behavior_tokens, torch.zeros(batch_size, 1, dtype=torch.long, device = input_id_sequence.device)), dim=1)
        repeat_behavior_tokens[:, behavior_indices] = 0
        repeat_behavior_tokens[repeat_behavior_tokens == self.eos] = 0
        return position_index, repeat_behavior_tokens

class PBADecoderRouter(nn.Module):
    def __init__(self, num_items, config: PBATransformerConfig):
        super().__init__()
        self.num_items = num_items
        self.num_experts = config.num_experts
        self.num_positions = config.num_positions
        self.num_behavior = config.num_behavior
        self.eos = config.eos_token_id
        self.pad = config.pad_token_id
        if config.Moe_behavior_only:
            self.pre_generated_position_index = torch.cat([(torch.tensor([0, 1, 1, 1], dtype=torch.long) + 1).repeat(self.num_items),torch.zeros(1, dtype=torch.long)])
        else:
            self.pre_generated_position_index = torch.cat([(torch.arange(self.num_positions, dtype=torch.long) + 1).repeat(self.num_items),torch.zeros(1, dtype=torch.long)])
        
        
    def forward(self, input_id_sequence: torch.Tensor) -> Tuple:
        batch_size, seq_length = input_id_sequence.size()
        position_index = self.pre_generated_position_index.to(input_id_sequence.device)[:seq_length].repeat(batch_size, 1)
        if seq_length == 1:
            behavior_indices = torch.zeros(batch_size, 1, dtype=torch.long, device = input_id_sequence.device)
        else:
            if not self.training:
                behavior_ids = input_id_sequence[:, 1]
                out_of_range = (behavior_ids < 1) | (behavior_ids > 4)
                behavior_ids[out_of_range] = 1
                input_id_sequence[:, 1] = behavior_ids
            behavior_indices = torch.zeros(batch_size, 1, dtype=torch.long, device = input_id_sequence.device)
            behavior_indices = torch.cat((behavior_indices, (torch.repeat_interleave(input_id_sequence[:,1].unsqueeze(1), seq_length - 1, dim=1))), dim=1)
        return position_index, behavior_indices

# Copied from transformers.models.t5.modeling_t5.T5DenseActDense with T5->SwitchTransformers
class PBATransformersDenseActDense(nn.Module):
    def __init__(self, config: PBATransformerConfig, behavior_injection = False):
        super().__init__()
        if behavior_injection:
            self.wi = nn.Linear((config.d_model + config.behavior_embedding_dim), config.d_ff, bias=False)
        else:
            self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states

class PBATransformersDenseActDenseHalfOutput(nn.Module):
    def __init__(self, config: PBATransformerConfig, behavior_injection = False):
        super().__init__()
        if behavior_injection:
            self.wi = nn.Linear((config.d_model + config.behavior_embedding_dim), config.d_ff, bias=False)
        else:
            self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model//2, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states

class PBATransformersSparseMLP(nn.Module):
    r"""
    Implementation of the Switch Transformers Sparse MLP module.
    """

    def __init__(self, config: PBATransformerConfig, expert_class: nn.Module = PBATransformersDenseActDense, is_sparse = False, behavior_injection = False):
        super().__init__()
        self.is_sparse = is_sparse
        if config.shared_expert and is_sparse:
            expert_class = PBATransformersDenseActDenseHalfOutput
            self.shared_expert = PBATransformersDenseActDenseHalfOutput(config, behavior_injection)
        self.has_shared_expert = config.shared_expert
        if self.is_sparse:
            self.experts = nn.ModuleDict()
            for idx in range(config.num_experts):
                self.experts[f"expert_{idx}"] = expert_class(config, behavior_injection)
        else:
            self.mlp = expert_class(config, behavior_injection)
        self.behavior_injection = behavior_injection
        if self.behavior_injection:
            self.behavior_embedding = nn.Embedding(config.num_behavior + 1, config.behavior_embedding_dim)

    def forward(self, hidden_states, position_index, behavior_index):
        r"""
        Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:

        1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`
        and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the
        hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).

        2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
        expert the corresponding hidden states.

        """
        if not self.has_shared_expert:
            next_states = torch.zeros_like(hidden_states)
            if self.behavior_injection:
                behavior_embedding = self.behavior_embedding(behavior_index)
                hidden_states = torch.cat((hidden_states, behavior_embedding), dim=-1)
            if self.is_sparse:
                for idx, expert in enumerate(self.experts.values()):
                    token_indices = (position_index == idx)
                    next_states[token_indices] = expert(hidden_states[token_indices]).to(next_states.dtype)
            else:
                next_states = self.mlp(hidden_states)

            return next_states
        else:
            next_states = torch.zeros((hidden_states.size(0), hidden_states.size(1), hidden_states.size(2)//2), device=hidden_states.device, dtype=hidden_states.dtype)
            if self.behavior_injection:
                behavior_embedding = self.behavior_embedding(behavior_index)
                hidden_states = torch.cat((hidden_states, behavior_embedding), dim=-1)
            if self.is_sparse:
                for idx, expert in enumerate(self.experts.values()):
                    token_indices = (position_index == idx)
                    next_states[token_indices] = expert(hidden_states[token_indices]).to(next_states.dtype)
                shared_next_states = self.shared_expert(hidden_states)
                next_states = torch.cat((next_states, shared_next_states), dim=-1)
                return next_states
            else:
                next_states = self.mlp(hidden_states)

            return next_states
    

class PBATransformersLayerFF(nn.Module):
    r"""
    Switch Transformers Feed Forward layer module. This is a wrapper around the Mixture of Experts module.

    Parameters:
        config : ([`SwitchTransformersConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        is_sparse (`bool`):
            Whether the MLP layer is a `Sparse` layer (contains a Mixture of Experts) or not
    """

    def __init__(self, config: PBATransformerConfig, is_sparse=False, behavior_injection=False):
        super().__init__()
        self.is_sparse = is_sparse

        # Check if it is a sparse layer, if not then it is a dense layer
        self.mlp = PBATransformersSparseMLP(config, is_sparse=self.is_sparse, behavior_injection=behavior_injection)

        self.layer_norm = SwitchTransformersLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, position_index, behavior_index):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.mlp(forwarded_states, position_index, behavior_index)

        output = hidden_states + self.dropout(forwarded_states)

        return output


class PBATransformersBlock(nn.Module):
    def __init__(self, config: PBATransformerConfig, has_relative_attention_bias=False, is_sparse=False, behavior_injection=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.is_sparse = is_sparse
        self.layer = nn.ModuleList()
        self.layer.append(
            SwitchTransformersLayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        )
        if self.is_decoder:
            self.layer.append(SwitchTransformersLayerCrossAttention(config))

        self.layer.append(PBATransformersLayerFF(config, is_sparse=self.is_sparse, behavior_injection=behavior_injection))

    def forward(
        self,
        hidden_states,
        position_indices,
        behavior_indices,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        output_router_logits=True,
        return_dict=True,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states, position_indices, behavior_indices)

        if isinstance(hidden_states, tuple):
            hidden_states, router_tuple = hidden_states
        else:
            router_tuple = (torch.zeros((1,), device=hidden_states.device, dtype=torch.int64),)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs + (router_tuple,)
        else:
            outputs = outputs + attention_outputs + (router_tuple,)

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights), (router_tuple)


class PBATransformersPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PBATransformerConfig
    base_model_prefix = "switch_transformers"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SwitchTransformersBlock"]

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, SwitchTransformersLayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(
            module,
            (PBATransformersForConditionalGeneration),
        ):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, PBATransformersDenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, SwitchTransformersAttention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))
        elif isinstance(module, PBATransformersSparseMLP):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            if module.is_sparse:
                for idx in range(self.config.num_experts):
                    module.experts[f"expert_{idx}"].wi.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
                    module.experts[f"expert_{idx}"].wo.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            else:
                module.mlp.wi.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
                module.mlp.wo.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
        
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In SwitchTransformers it is usually set"
                " to the pad_token_id. See SwitchTransformers docs for more information"
            )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

class PBATransformersStack(PBATransformersPreTrainedModel):
    def __init__(self, config: PBATransformerConfig, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.router = PBADecoderRouter(1, config)
        else:
            self.router = PBAEncoderRouter(config.n_positions, config)
        self.sparse_layers = config.sparse_layers_decoder if self.is_decoder else config.sparse_layers_encoder
        behavior_injection_layers = config.behavior_injection_decoder if self.is_decoder else config.behavior_injection_encoder
        config.num_layers = config.num_decoder_layers if self.is_decoder else config.num_layers
        self.block = nn.ModuleList()
        for i in range(config.num_layers):
            is_sparse = i in self.sparse_layers
            is_injection = i in behavior_injection_layers
            self.block.append(
                PBATransformersBlock(config, has_relative_attention_bias=bool(i == 0), is_sparse=is_sparse, behavior_injection=is_injection)
            )

        self.final_layer_norm = SwitchTransformersLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()

        self.device_map = None
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_router_logits=True,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        position_indices, behavior_indices = self.router(input_ids)
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_router_probs = () if output_router_logits else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    hidden_states,
                    position_indices,
                    behavior_indices,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    position_indices,
                    behavior_indices,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                )

            router_probs = layer_outputs[-1]
            layer_outputs = layer_outputs[:-1]

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            if output_router_logits:
                all_router_probs = all_router_probs + (router_probs,)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                    all_router_probs,
                ]
                if v is not None
            )
        return MoEModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            router_probs=all_router_probs,
        )


class PBATransformersForConditionalGeneration(PBATransformersPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: PBATransformerConfig):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = PBATransformersStack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = PBATransformersStack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.router_z_loss_coef = config.router_z_loss_coef
        self.router_aux_loss_coef = config.router_aux_loss_coef
        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.device_map = None

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = True,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqMoEOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
        >>> model = SwitchTransformersForConditionalGeneration.from_pretrained("google/switch-base-8")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> # . To, let’s say you have a dog. To summarize:
        >>> # Since the model has been trained on MLM, this will output gibberish
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_router_logits=output_router_logits,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, MoEModelOutput):
            try:
                encoder_outputs = MoEModelOutput(
                    last_hidden_state=encoder_outputs[0],
                    hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                    attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                    router_probs=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
                )
            except:
                encoder_outputs = MoEModelOutput(
                    last_hidden_state=encoder_outputs['last_hidden_state'],
                    hidden_states= None,
                    attentions=None,
                    router_probs=None,
                )

        hidden_states = encoder_outputs[0]
        #hidden_states = encoder_outputs['last_hidden_state']
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        encoder_z_loss = 0
        encoder_aux_loss = 0
        decoder_z_loss = 0
        decoder_aux_loss = 0

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            if output_router_logits:
                z_loss = self.router_z_loss_coef * (encoder_z_loss + decoder_z_loss)
                aux_loss = self.router_aux_loss_coef * (encoder_aux_loss + decoder_aux_loss)
                loss = loss + z_loss + aux_loss

        if not return_dict:
            output = (lm_logits,)
            if output_router_logits:
                output += (encoder_z_loss, encoder_aux_loss, decoder_z_loss, decoder_aux_loss)
            output += (*decoder_outputs[1:], *encoder_outputs)

            return ((loss,) + output) if loss is not None else output

        return Seq2SeqMoEOutput(
            loss=loss,
            logits=lm_logits,
            encoder_z_loss=encoder_z_loss,
            encoder_aux_loss=encoder_aux_loss,
            decoder_z_loss=decoder_z_loss,
            decoder_aux_loss=decoder_aux_loss,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            decoder_router_logits=decoder_outputs.router_probs,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            encoder_router_logits=encoder_outputs.router_probs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    "expected reordered_layer_past_states to have the same shape than layer_past_states, "
                    f"but got {reordered_layer_past_states[0].shape} and {layer_past_states[0].shape}"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    "expected layer_past_states to have the same length as reordered_layer_past_states, "
                    f"but got {len(layer_past_states)} and {len(reordered_layer_past_states)}"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past



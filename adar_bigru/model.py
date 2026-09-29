from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .ops import BOS, COPY, DELETE, EOS, INS_A, INS_C, PAD, SUB_A, SUB_C


@dataclass
class ModelConfig:
    d_model: int = 128
    nhead: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    feedforward: int = 384
    dropout: float = 0.1
    max_length: int = 256
    architecture: str = "transformer_transformer"

    def to_dict(self):
        return asdict(self)


class FiLM(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.affine = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, hidden, condition):
        gamma, beta = self.affine(condition).chunk(2, dim=-1)
        return hidden * (1 + gamma[:, None, :]) + beta[:, None, :]


class RankConditionedEditTransformer(nn.Module):
    """Template-edit Transformer with score/rank token and layer-wise FiLM conditioning."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.source_embedding = nn.Embedding(8, d, padding_idx=PAD)
        self.template_embedding = nn.Embedding(7, d, padding_idx=PAD)
        self.action_embedding = nn.Embedding(13, d, padding_idx=PAD)
        self.position_embedding = nn.Embedding(config.max_length + 2, d)
        self.score_bucket_embedding = nn.Embedding(22, d)
        self.rank_bucket_embedding = nn.Embedding(12, d)
        self.condition_mlp = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.condition_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.condition_token, std=0.02)
        self.dropout = nn.Dropout(config.dropout)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, config.nhead, config.feedforward, config.dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(config.encoder_layers)
        ])
        self.encoder_film = nn.ModuleList([FiLM(d) for _ in range(config.encoder_layers)])
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d, config.nhead, config.feedforward, config.dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(config.decoder_layers)
        ])
        self.decoder_film = nn.ModuleList([FiLM(d) for _ in range(config.decoder_layers)])
        self.norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, 13)
        self.quality_head = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(d, 1)
        )

    def condition(self, score_percent, percentile, conditioned=True):
        if conditioned:
            score_bucket = torch.clamp((score_percent / 5).long(), 0, 20)
            rank_bucket = torch.clamp((percentile * 10).long(), 0, 10)
            continuous = torch.stack((score_percent / 100.0, percentile), dim=-1)
        else:
            score_bucket = torch.full_like(score_percent.long(), 21)
            rank_bucket = torch.full_like(score_percent.long(), 11)
            continuous = torch.zeros((len(score_percent), 2), dtype=score_percent.dtype,
                                     device=score_percent.device)
        return (self.score_bucket_embedding(score_bucket)
                + self.rank_bucket_embedding(rank_bucket)
                + self.condition_mlp(continuous))

    def encode(self, source, score_percent, percentile, conditioned=True):
        source_pad = source.eq(PAD)
        positions = torch.arange(source.size(1), device=source.device)[None, :]
        context = self.condition(score_percent, percentile, conditioned)
        x = self.source_embedding(source) + self.position_embedding(positions)
        condition_token = self.condition_token.expand(source.size(0), -1, -1) + context[:, None, :]
        x = torch.cat((condition_token, x), dim=1)
        memory_pad = torch.cat((torch.zeros((source.size(0), 1), dtype=torch.bool, device=source.device),
                                source_pad), dim=1)
        for layer, film in zip(self.encoder_layers, self.encoder_film, strict=True):
            x = layer(self.dropout(x), src_key_padding_mask=memory_pad)
            x = film(x, context)
        return self.norm(x), memory_pad, context

    def decode_states(self, memory, memory_pad, context, template, decoder_input, pointers):
        if decoder_input.size(1) > self.config.max_length + 1:
            raise ValueError("Decoder input exceeds configured max_length")
        extended_template = torch.cat((template, torch.zeros(template.size(0), 1,
                                      dtype=template.dtype, device=template.device)), dim=1)
        current_base = torch.gather(extended_template, 1, pointers.clamp(max=template.size(1)))
        x = (self.action_embedding(decoder_input) + self.position_embedding(pointers)
             + self.template_embedding(current_base) + context[:, None, :])
        length = decoder_input.size(1)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=decoder_input.device), diagonal=1)
        target_pad = decoder_input.eq(PAD)
        for layer, film in zip(self.decoder_layers, self.decoder_film, strict=True):
            x = layer(self.dropout(x), memory, tgt_mask=causal,
                      tgt_key_padding_mask=target_pad, memory_key_padding_mask=memory_pad)
            x = film(x, context)
        return self.norm(x)

    def forward(self, source, template, decoder_input, pointers, score_percent, percentile,
                conditioned=True):
        memory, memory_pad, context = self.encode(source, score_percent, percentile, conditioned)
        states = self.decode_states(memory, memory_pad, context, template, decoder_input, pointers)
        return self.output(states)

    def predict_quality(self, source, actions):
        source_positions = torch.arange(source.size(1), device=source.device)[None, :]
        action_positions = torch.arange(actions.size(1), device=actions.device)[None, :]
        source_hidden = self.source_embedding(source) + self.position_embedding(source_positions)
        action_hidden = self.action_embedding(actions) + self.position_embedding(action_positions)
        source_mask = source.ne(PAD)
        action_mask = actions.ne(PAD)
        source_pool = (source_hidden * source_mask[:, :, None]).sum(1) / source_mask.sum(1, keepdim=True).clamp_min(1)
        action_pool = (action_hidden * action_mask[:, :, None]).sum(1) / action_mask.sum(1, keepdim=True).clamp_min(1)
        return torch.sigmoid(self.quality_head(torch.cat((source_pool, action_pool), dim=-1)).squeeze(-1))

    @torch.inference_mode()
    def generate_actions(self, source, template, score_percent, percentile, max_edits: int,
                         temperature=0.8, top_k=6, guidance_scale=1.0):
        self.eval()
        batch_size = source.size(0)
        lengths = source.ne(PAD).sum(dim=1)
        memory, memory_pad, context = self.encode(source, score_percent, percentile, True)
        if guidance_scale != 1:
            memory_u, memory_pad_u, context_u = self.encode(source, score_percent, percentile, False)
        actions = torch.full((batch_size, 1), BOS, dtype=torch.long, device=source.device)
        pointers = torch.zeros((batch_size, 1), dtype=torch.long, device=source.device)
        edits = torch.zeros(batch_size, dtype=torch.long, device=source.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=source.device)
        max_steps = int(lengths.max().item()) + max_edits + 1
        for _ in range(max_steps):
            states = self.decode_states(memory, memory_pad, context, template, actions, pointers)
            logits = self.output(states[:, -1]).float()
            if guidance_scale != 1:
                states_u = self.decode_states(memory_u, memory_pad_u, context_u, template, actions, pointers)
                unconditional = self.output(states_u[:, -1]).float()
                logits = unconditional + guidance_scale * (logits - unconditional)
            at_end = pointers[:, -1] >= lengths
            logits[:, :EOS] = torch.finfo(logits.dtype).min
            logits[:, EOS] = logits[:, EOS].masked_fill(~at_end, torch.finfo(logits.dtype).min)
            logits[:, COPY:INS_A] = logits[:, COPY:INS_A].masked_fill(
                at_end[:, None], torch.finfo(logits.dtype).min)
            exhausted = edits >= max_edits
            logits[:, DELETE:INS_C + 1] = logits[:, DELETE:INS_C + 1].masked_fill(
                exhausted[:, None], torch.finfo(logits.dtype).min)
            if temperature <= 0:
                next_action = logits.argmax(dim=1)
            else:
                scaled = logits / temperature
                if top_k:
                    cutoff = scaled.topk(min(top_k, scaled.size(1)), dim=1).values[:, -1, None]
                    scaled = scaled.masked_fill(scaled < cutoff, torch.finfo(scaled.dtype).min)
                next_action = torch.multinomial(torch.softmax(scaled, dim=1), 1).squeeze(1)
            next_action = torch.where(finished, torch.full_like(next_action, EOS), next_action)
            consuming = ((next_action == COPY) | (next_action == DELETE)
                         | ((next_action >= SUB_A) & (next_action <= SUB_C)))
            edit_action = ((next_action == DELETE)
                           | ((next_action >= SUB_A) & (next_action <= INS_C)))
            edits += edit_action & ~finished
            new_pointer = pointers[:, -1] + (consuming & ~finished)
            actions = torch.cat((actions, next_action[:, None]), dim=1)
            pointers = torch.cat((pointers, new_pointer[:, None]), dim=1)
            finished |= next_action.eq(EOS)
            if finished.all():
                break
        if not finished.all():
            raise RuntimeError("Constrained decoder failed to reach EOS")
        return actions[:, 1:], edits

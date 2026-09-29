from __future__ import annotations

import torch
from torch import nn

from .model import FiLM, ModelConfig, RankConditionedEditTransformer
from .ops import BOS, COPY, DELETE, EOS, INS_A, INS_C, PAD, SUB_A, SUB_C


ARCHITECTURES = (
    "transformer_transformer",
    "bigru_gru",
    "bilstm_lstm",
    "transformer_gru",
    "bigru_transformer",
    "tcn_gru",
    "rnn_rnn",
)


class ResidualTCNBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(width, width, kernel_size=3, padding=dilation,
                              dilation=dilation)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value):
        update = self.conv(value.transpose(1, 2)).transpose(1, 2)
        return self.norm(value + self.dropout(torch.nn.functional.gelu(update)))


class AlternativeEditModel(nn.Module):
    """Edit-action model with interchangeable sequence encoder and decoder."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.architecture not in ARCHITECTURES[1:]:
            raise ValueError(f"Unsupported alternative architecture: {config.architecture}")
        self.config = config
        d = config.d_model
        if d % 2:
            raise ValueError("Recurrent bidirectional encoders require an even d_model")
        self.source_embedding = nn.Embedding(8, d, padding_idx=PAD)
        self.template_embedding = nn.Embedding(7, d, padding_idx=PAD)
        self.action_embedding = nn.Embedding(13, d, padding_idx=PAD)
        self.position_embedding = nn.Embedding(config.max_length + 2, d)
        self.score_bucket_embedding = nn.Embedding(22, d)
        self.rank_bucket_embedding = nn.Embedding(12, d)
        self.condition_mlp = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.dropout = nn.Dropout(config.dropout)
        self.encoder_kind, self.decoder_kind = self._kinds(config.architecture)
        self.encoder_film = None
        self.decoder_film = None

        if self.encoder_kind == "transformer":
            self.encoder = nn.ModuleList([
                nn.TransformerEncoderLayer(d, config.nhead, config.feedforward,
                                           config.dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
                for _ in range(config.encoder_layers)
            ])
            self.encoder_films = nn.ModuleList([FiLM(d) for _ in range(config.encoder_layers)])
        elif self.encoder_kind == "tcn":
            self.encoder = nn.ModuleList([
                ResidualTCNBlock(d, 2 ** index, config.dropout)
                for index in range(config.encoder_layers + 2)
            ])
            self.encoder_films = nn.ModuleList([FiLM(d) for _ in self.encoder])
        else:
            cell = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}[self.encoder_kind]
            self.encoder = cell(d, d // 2, num_layers=config.encoder_layers,
                                dropout=config.dropout if config.encoder_layers > 1 else 0,
                                bidirectional=True, batch_first=True)
            state_size = config.encoder_layers * 2 * (d // 2)
            self.encoder_h0 = nn.Linear(d, state_size)
            self.encoder_c0 = nn.Linear(d, state_size) if self.encoder_kind == "lstm" else None
            self.encoder_film = FiLM(d)

        if self.decoder_kind == "transformer":
            self.decoder = nn.ModuleList([
                nn.TransformerDecoderLayer(d, config.nhead, config.feedforward,
                                           config.dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
                for _ in range(config.decoder_layers)
            ])
            self.decoder_films = nn.ModuleList([FiLM(d) for _ in range(config.decoder_layers)])
        else:
            cell = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}[self.decoder_kind]
            self.decoder = cell(d, d, num_layers=config.decoder_layers,
                                dropout=config.dropout if config.decoder_layers > 1 else 0,
                                batch_first=True)
            self.decoder_h0 = nn.Linear(2 * d, config.decoder_layers * d)
            self.decoder_c0 = (nn.Linear(2 * d, config.decoder_layers * d)
                               if self.decoder_kind == "lstm" else None)
            self.attention_query = nn.Linear(d, d)
            self.attention_key = nn.Linear(d, d)
            self.decoder_context = nn.Linear(2 * d, d)
            self.decoder_fusion = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
            self.decoder_film = FiLM(d)
        self.norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, 13)
        self.quality_head = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(d, 1)
        )

    @staticmethod
    def _kinds(architecture):
        return {
            "bigru_gru": ("gru", "gru"),
            "bilstm_lstm": ("lstm", "lstm"),
            "transformer_gru": ("transformer", "gru"),
            "bigru_transformer": ("gru", "transformer"),
            "tcn_gru": ("tcn", "gru"),
            "rnn_rnn": ("rnn", "rnn"),
        }[architecture]

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
        value = self.source_embedding(source) + self.position_embedding(positions) + context[:, None, :]
        if self.encoder_kind == "transformer":
            for layer, film in zip(self.encoder, self.encoder_films, strict=True):
                value = layer(self.dropout(value), src_key_padding_mask=source_pad)
                value = film(value, context)
        elif self.encoder_kind == "tcn":
            for layer, film in zip(self.encoder, self.encoder_films, strict=True):
                value = film(layer(self.dropout(value)), context)
                value = value.masked_fill(source_pad[:, :, None], 0)
        else:
            layers = self.config.encoder_layers
            directions = 2
            hidden = self.config.d_model // 2
            h0 = self.encoder_h0(context).view(source.size(0), layers * directions, hidden).transpose(0, 1).contiguous()
            state = h0
            if self.encoder_kind == "lstm":
                c0 = self.encoder_c0(context).view(source.size(0), layers * directions, hidden).transpose(0, 1).contiguous()
                state = (h0, c0)
            # Direct padded execution keeps the recurrent kernels on the fast
            # cuDNN path. Padding inputs are zeroed and masked from attention.
            value = value.masked_fill(source_pad[:, :, None], 0)
            value, _ = self.encoder(self.dropout(value), state)
            value = value.masked_fill(source_pad[:, :, None], 0)
            value = self.encoder_film(value, context)
        return self.norm(value), source_pad, context

    @staticmethod
    def _pool(memory, memory_pad):
        mask = ~memory_pad
        return (memory * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)

    def decode_states(self, memory, memory_pad, context, template, decoder_input, pointers):
        extended_template = torch.cat((template, torch.zeros(template.size(0), 1,
                                      dtype=template.dtype, device=template.device)), dim=1)
        current_base = torch.gather(extended_template, 1,
                                    pointers.clamp(max=template.size(1)))
        value = (self.action_embedding(decoder_input) + self.position_embedding(pointers)
                 + self.template_embedding(current_base) + context[:, None, :])
        if self.decoder_kind == "transformer":
            length = decoder_input.size(1)
            causal = torch.triu(torch.ones(length, length, dtype=torch.bool,
                                           device=decoder_input.device), diagonal=1)
            target_pad = decoder_input.eq(PAD)
            for layer, film in zip(self.decoder, self.decoder_films, strict=True):
                value = layer(self.dropout(value), memory, tgt_mask=causal,
                              tgt_key_padding_mask=target_pad,
                              memory_key_padding_mask=memory_pad)
                value = film(value, context)
        else:
            pooled = self._pool(memory, memory_pad)
            query = self.attention_query(context)[:, None, :]
            keys = self.attention_key(memory)
            attention_logits = (query * keys).sum(-1) / (self.config.d_model ** 0.5)
            attention_logits = attention_logits.masked_fill(memory_pad,
                                                            torch.finfo(attention_logits.dtype).min)
            attention_weights = torch.softmax(attention_logits, dim=1)
            attended = (memory * attention_weights[:, :, None]).sum(1)
            initial = torch.cat((attended, context), dim=-1)
            layers = self.config.decoder_layers
            h0 = self.decoder_h0(initial).view(value.size(0), layers,
                                                    self.config.d_model).transpose(0, 1).contiguous()
            state = h0
            if self.decoder_kind == "lstm":
                c0 = self.decoder_c0(initial).view(value.size(0), layers,
                                                      self.config.d_model).transpose(0, 1).contiguous()
                state = (h0, c0)
            value = value + self.decoder_context(initial)[:, None, :]
            recurrent, _ = self.decoder(self.dropout(value), state)
            expanded_context = attended[:, None, :].expand(-1, recurrent.size(1), -1)
            value = self.decoder_fusion(torch.cat((recurrent, expanded_context), dim=-1))
            value = self.decoder_film(value, context)
        return self.norm(value)

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
            memory_u, memory_pad_u, context_u = self.encode(source, score_percent,
                                                           percentile, False)
        actions = torch.full((batch_size, 1), BOS, dtype=torch.long, device=source.device)
        pointers = torch.zeros((batch_size, 1), dtype=torch.long, device=source.device)
        edits = torch.zeros(batch_size, dtype=torch.long, device=source.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=source.device)
        max_steps = int(lengths.max().item()) + max_edits + 1
        for _ in range(max_steps):
            states = self.decode_states(memory, memory_pad, context, template, actions, pointers)
            logits = self.output(states[:, -1]).float()
            if guidance_scale != 1:
                states_u = self.decode_states(memory_u, memory_pad_u, context_u,
                                              template, actions, pointers)
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
                    scaled = scaled.masked_fill(scaled < cutoff,
                                                torch.finfo(scaled.dtype).min)
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


def build_model(config: ModelConfig):
    if config.architecture == "transformer_transformer":
        return RankConditionedEditTransformer(config)
    return AlternativeEditModel(config)

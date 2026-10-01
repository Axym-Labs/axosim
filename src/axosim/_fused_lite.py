"""Installed fused Lite neuron step, extracted from validated iteration 572.

This kernel fuses recurrence and decoding only. Connected graph routing and
native scheduling are supplied by connected_population, not the historical
specialized million-neuron router. Adaptation values are passed directly with
unit scales; the facade does not silently quantize caller coefficients.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _adaptive_support_step_kernel(
    token,
    previous_state,
    adaptation,
    adaptation_scales,
    input_weight,
    recurrent_weight,
    state_bias,
    shared_decay_logit,
    output_weight,
    skip_weight,
    output_bias,
    next_state,
    output,
    population: tl.constexpr,
    token_width: tl.constexpr,
    state_width: tl.constexpr,
    adaptation_width: tl.constexpr,
    outputs: tl.constexpr,
    block_rows: tl.constexpr,
    block_token: tl.constexpr,
    block_state: tl.constexpr,
    block_outputs: tl.constexpr,
):
    rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    token_index = tl.arange(0, block_token)
    state_index = tl.arange(0, block_state)
    output_index = tl.arange(0, block_outputs)
    row_mask = rows < population
    token_mask = token_index < token_width
    state_mask = state_index < state_width
    output_mask = output_index < outputs
    token_matrix_mask = row_mask[:, None] & token_mask[None, :]
    state_matrix_mask = row_mask[:, None] & state_mask[None, :]
    adaptation_base = rows[:, None] * adaptation_width

    raw_token = tl.load(
        token + rows[:, None] * token_width + token_index[None, :],
        mask=token_matrix_mask,
        other=0.0,
    ).to(tl.float32)
    token_offset = tl.load(
        adaptation + adaptation_base + token_index[None, :],
        mask=token_matrix_mask,
        other=0,
    ).to(tl.float32) * tl.load(adaptation_scales)
    token_gain = tl.load(
        adaptation + adaptation_base + token_width + token_index[None, :],
        mask=token_matrix_mask,
        other=0,
    ).to(tl.float32) * tl.load(adaptation_scales + 1)
    adapted_token = ((raw_token + token_offset) * (1.0 + token_gain)).to(tl.float16)

    previous = tl.load(
        previous_state + rows[:, None] * state_width + state_index[None, :],
        mask=state_matrix_mask,
        other=0.0,
    ).to(tl.float32)
    input_matrix = tl.load(
        input_weight + state_index[:, None] * token_width + token_index[None, :],
        mask=state_mask[:, None] & token_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    recurrent_matrix = tl.load(
        recurrent_weight + state_index[:, None] * state_width + state_index[None, :],
        mask=state_mask[:, None] & state_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    forcing = tl.dot(adapted_token, tl.trans(input_matrix))
    recurrence = tl.dot(
        previous.to(tl.float16),
        tl.trans(recurrent_matrix),
    )
    state_adaptation_base = 2 * token_width
    decay_delta = tl.load(
        adaptation + adaptation_base + state_adaptation_base + state_index[None, :],
        mask=state_matrix_mask,
        other=0,
    ).to(tl.float32) * tl.load(adaptation_scales + 2)
    state_gain = tl.load(
        adaptation
        + adaptation_base
        + state_adaptation_base
        + state_width
        + state_index[None, :],
        mask=state_matrix_mask,
        other=0,
    ).to(tl.float32) * tl.load(adaptation_scales + 3)
    state_offset = tl.load(
        adaptation
        + adaptation_base
        + state_adaptation_base
        + 2 * state_width
        + state_index[None, :],
        mask=state_matrix_mask,
        other=0,
    ).to(tl.float32) * tl.load(adaptation_scales + 4)
    proposal_argument = (
        forcing
        + recurrence
        + tl.load(
            state_bias + state_index,
            mask=state_mask,
            other=0.0,
        )[None, :]
        + state_offset
    )
    proposal = 2.0 * tl.sigmoid(2.0 * proposal_argument) - 1.0
    decay = tl.sigmoid(
        tl.load(
            shared_decay_logit + state_index,
            mask=state_mask,
            other=0.0,
        )[None, :]
        + decay_delta
    )
    state = decay * previous + (1.0 - decay) * proposal * (1.0 + state_gain)
    tl.store(
        next_state + rows[:, None] * state_width + state_index[None, :],
        state,
        mask=state_matrix_mask,
    )

    state_to_output = tl.load(
        output_weight + output_index[:, None] * state_width + state_index[None, :],
        mask=output_mask[:, None] & state_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    token_to_output = tl.load(
        skip_weight + output_index[:, None] * token_width + token_index[None, :],
        mask=output_mask[:, None] & token_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    decoded = tl.dot(
        state.to(tl.float16),
        tl.trans(state_to_output),
    )
    decoded += tl.dot(
        adapted_token,
        tl.trans(token_to_output),
    )
    decoded += tl.load(
        output_bias + output_index,
        mask=output_mask,
        other=0.0,
    )[None, :]
    output_adaptation_base = state_adaptation_base + 3 * state_width
    output_gain = tl.load(
        adaptation + adaptation_base + output_adaptation_base + output_index[None, :],
        mask=row_mask[:, None] & output_mask[None, :],
        other=0,
    ).to(tl.float32) * tl.load(adaptation_scales + 5)
    output_offset = tl.load(
        adaptation
        + adaptation_base
        + output_adaptation_base
        + outputs
        + output_index[None, :],
        mask=row_mask[:, None] & output_mask[None, :],
        other=0,
    ).to(tl.float32) * tl.load(adaptation_scales + 6)
    decoded = decoded * (1.0 + output_gain) + output_offset
    tl.store(
        output + rows[:, None] * outputs + output_index[None, :],
        decoded,
        mask=row_mask[:, None] & output_mask[None, :],
    )


def fused_lite_step(model, token, state, coefficients):
    """Return ``(next_state, forecasts)`` for contiguous FP16 CUDA tensors.

    ``forecasts`` has shape ``(N, 4, 2)`` in native model coordinates.
    Coefficients are direct packed runtime values, not quantized integer codes.
    """
    n, token_width = token.shape
    state_width = state.shape[1]
    next_state = torch.empty_like(state)
    output = torch.empty((n, 8), dtype=token.dtype, device=token.device)
    scales = torch.ones(7, dtype=torch.float32, device=token.device)
    _adaptive_support_step_kernel[(triton.cdiv(n, 32),)](
        token,
        state,
        coefficients,
        scales,
        model.support_input.weight,
        model.support_recurrent.weight,
        model.state_bias,
        model.shared_decay_logit,
        model.output_head.weight,
        model.skip_head.weight,
        model.output_head.bias,
        next_state,
        output,
        population=n,
        token_width=token_width,
        state_width=state_width,
        adaptation_width=model.cache_width,
        outputs=8,
        block_rows=32,
        block_token=max(16, triton.next_power_of_2(token_width)),
        block_state=max(16, triton.next_power_of_2(state_width)),
        block_outputs=16,
    )
    return next_state, output.reshape(n, 4, 2)

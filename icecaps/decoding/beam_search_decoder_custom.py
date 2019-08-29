from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf

from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import nest


class BeamSearchDecoder(beam_search_decoder.BeamSearchDecoder):
    """BeamSearch sampling decoder.

            **NOTE** If you are using the `BeamSearchDecoder` with a cell wrapped in
            `AttentionWrapper`, then you must ensure that:

            - The encoder output has been tiled to `beam_width` via
                    @{tf.contrib.seq2seq.tile_batch} (NOT `tf.tile`).
            - The `batch_size` argument passed to the `zero_state` method of this
                    wrapper is equal to `true_batch_size * beam_width`.
            - The initial state created with `zero_state` above contains a
                    `cell_state` value containing properly tiled final state from the
                    encoder.

            An example:

            ```
            tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
                            encoder_outputs, multiplier=beam_width)
            tiled_encoder_final_state = tf.conrib.seq2seq.tile_batch(
                            encoder_final_state, multiplier=beam_width)
            tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
                            sequence_length, multiplier=beam_width)
            attention_mechanism = MyFavoriteAttentionMechanism(
                            num_units=attention_depth,
                            memory=tiled_inputs,
                            memory_sequence_length=tiled_sequence_length)
            attention_cell = AttentionWrapper(cell, attention_mechanism, ...)
            decoder_initial_state = attention_cell.zero_state(
                            dtype, batch_size=true_batch_size * beam_width)
            decoder_initial_state = decoder_initial_state.clone(
                            cell_state=tiled_encoder_final_state)
            ```
    """

    def __init__(self,
                 cell,
                 embedding,
                 start_tokens,
                 end_token,
                 initial_state,
                 beam_width,
                 output_layer=None,
                 length_penalty_weight=0.0,
                 coverage_penalty_weight=0.0,
                 reorder_tensor_arrays=True,
                 skip_tokens_decoding=None,
                 shrink_vocab=0,
                 start_token_logits=None):
        """Initialize the BeamSearchDecoder.

        Args:
                cell: An `RNNCell` instance.
                embedding: A callable that takes a vector tensor of `ids` (argmax ids),
                        or the `params` argument for `embedding_lookup`.
                start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
                end_token: `int32` scalar, the token that marks end of decoding.
                initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
                beam_width:  Python integer, the number of beams.
                output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
                        `tf.layers.Dense`.  Optional layer to apply to the RNN output prior
                        to storing the result or sampling.
                length_penalty_weight: Float weight to penalize length. Disabled with 0.0.
                coverage_penalty_weight: Float weight to penalize the coverage of source
                        sentence. Disabled with 0.0.
                reorder_tensor_arrays: If `True`, `TensorArray`s' elements within the cell
                        state will be reordered according to the beam search path. If the
                        `TensorArray` can be reordered, the stacked form will be returned.
                        Otherwise, the `TensorArray` will be returned as is. Set this flag to
                        `False` if the cell state contains `TensorArray`s that are not amenable
                        to reordering.
                skip_tokens_decoding: A list of tokens that should be skipped while decoding.
                        Defaults to None, that is, not skipping any tokens.
                shrink_vocab: Use only top 'N' tokens while decoding. Disabled with 0
                start_token_logits: Logits for the start tokens.  Used if _GO tokens are not
                        used. Defaults to None.

        Raises:
                TypeError: if `cell` is not an instance of `RNNCell`,
                        or `output_layer` is not an instance of `tf.layers.Layer`.
                ValueError: If `start_tokens` is not a vector or
                        `end_token` is not a scalar.
                ValueError: If `start token logits` not provided and `use_go_tokens`
                        is set to be False.
        """
        rnn_cell_impl.assert_like_rnncell(
            "cell", cell)  # pylint: disable=protected-access
        if (output_layer is not None and not isinstance(output_layer, layers_base.Layer)):
            raise TypeError(
                "output_layer must be a Layer, received: %s" % type(output_layer))
        self._cell = cell
        self._output_layer = output_layer
        self._reorder_tensor_arrays = reorder_tensor_arrays
        self._use_go_tokens = True
        if not self._use_go_tokens and start_token_logits is None:
            raise ValueError(
                "start token logits must be provided if use_go_tokens is False")
        
        if callable(embedding):
            self._embedding_fn = embedding
        else:
            self._embedding_fn = (
                lambda ids: embedding_ops.embedding_lookup(embedding, ids))

        if self._use_go_tokens:
            self._start_tokens = ops.convert_to_tensor(
                start_tokens, dtype=dtypes.int32, name="start_tokens")
            if self._start_tokens.get_shape().ndims != 1:
                raise ValueError("start_tokens must be a vector")

        self._end_token = ops.convert_to_tensor(
            end_token, dtype=dtypes.int32, name="end_token")
        if self._end_token.get_shape().ndims != 0:
            raise ValueError("end_token must be a scalar")

        if self._use_go_tokens:
            self._batch_size = array_ops.size(start_tokens)
        else:
            self._batch_size = array_ops.size(start_tokens) // beam_width

        self._beam_width = beam_width
        self._length_penalty_weight = length_penalty_weight
        self._coverage_penalty_weight = coverage_penalty_weight
        self._initial_cell_state = nest.map_structure(
            self._maybe_split_batch_beams, initial_state, self._cell.state_size)

        if self._use_go_tokens:
            self._start_tokens = array_ops.tile(
                array_ops.expand_dims(self._start_tokens, 1), [1, self._beam_width])
        else:
            self._start_tokens = start_tokens

        self._start_inputs = self._embedding_fn(self._start_tokens)
        self._finished = array_ops.one_hot(
            array_ops.zeros([self._batch_size], dtype=dtypes.int32),
            depth=self._beam_width,
            on_value=False,
            off_value=True,
            dtype=dtypes.bool)

        self._skip_tokens_decoding = skip_tokens_decoding
        self._shrink_vocab = shrink_vocab
        self._raw_end_token = end_token
        self._start_token_logits = start_token_logits

    def initialize(self, name=None):
        """Initialize the decoder.
        Args:
                name: Name scope for any created operations.
        Returns:
        `(finished, start_inputs, initial_state)`.
        """
        finished, start_inputs = self._finished, self._start_inputs

        dtype = nest.flatten(self._initial_cell_state)[0].dtype

        if self._start_token_logits is None:
            log_probs = array_ops.one_hot(  # shape(batch_sz, beam_sz)
                array_ops.zeros([self._batch_size], dtype=dtypes.int32),
                depth=self._beam_width,
                on_value=ops.convert_to_tensor(0.0, dtype=dtype),
                off_value=ops.convert_to_tensor(-np.Inf, dtype=dtype),
                dtype=dtype)
        else:
            log_probs = self._start_token_logits

        sequence_lengths = array_ops.zeros(
            [self._batch_size, self._beam_width], dtype=dtypes.int64)

        # Start tokens are part of output if no _GO token used. Make changes accordingly
        if not self._use_go_tokens:
            finished = math_ops.equal(self._start_tokens, self._raw_end_token)

            sequence_lengths = array_ops.where(
                math_ops.logical_not(finished),
                array_ops.fill(array_ops.shape(sequence_lengths),
                               tf.constant(1, dtype=dtypes.int64)),
                sequence_lengths)

        init_attention_probs = beam_search_decoder.get_attention_probs(
            self._initial_cell_state, self._coverage_penalty_weight)
        if init_attention_probs is None:
            init_attention_probs = ()

        initial_state = beam_search_decoder.BeamSearchDecoderState(
            cell_state=self._initial_cell_state,
            log_probs=log_probs,
            finished=finished,
            lengths=sequence_lengths,
            accumulated_attention_probs=init_attention_probs)

        return (finished, start_inputs, initial_state)

    def step(self, time, inputs, state, name=None):
        """Perform a decoding step.

        Args:
                time: scalar `int32` tensor.
                inputs: A (structure of) input tensors.
                state: A (structure of) state tensors and TensorArrays.
                name: Name scope for any created operations.

        Returns:
                `(outputs, next_state, next_inputs, finished)`.
        """
        batch_size = self._batch_size
        beam_width = self._beam_width
        end_token = self._end_token
        length_penalty_weight = self._length_penalty_weight
        coverage_penalty_weight = self._coverage_penalty_weight

        with ops.name_scope(name, "BeamSearchDecoderStep", (time, inputs, state)):
            cell_state = state.cell_state
            inputs = nest.map_structure(
                lambda inp: self._merge_batch_beams(inp, s=inp.shape[2:]), inputs)
            cell_state = nest.map_structure(self._maybe_merge_batch_beams, cell_state,
                                            self._cell.state_size)
            cell_outputs, next_cell_state = self._cell(inputs, cell_state)
            cell_outputs = nest.map_structure(
                lambda out: self._split_batch_beams(out, out.shape[1:]), cell_outputs)
            next_cell_state = nest.map_structure(
                self._maybe_split_batch_beams, next_cell_state, self._cell.state_size)

            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)

            if self._shrink_vocab > 0:
                self._skip_tokens_decoding += list(
                    range(self._shrink_vocab, cell_outputs.get_shape()[2]))
                self._skip_tokens_decoding = sorted(
                    set(self._skip_tokens_decoding))
                # Never skip _END token, no matter what
                if self._raw_end_token in self._skip_tokens_decoding:
                    self._skip_tokens_decoding.remove(self._raw_end_token)

            # Assign least possible logit for given list of tokens to avoid those tokens while decoding
            if len(self._skip_tokens_decoding) > 0:

                token_num = cell_outputs.get_shape()[2]
                minimum_activation = tf.reduce_min(cell_outputs) - 1
                blacklist = tf.sparse_to_dense(
                    self._skip_tokens_decoding,
                    output_shape=[cell_outputs.get_shape()[2]],
                    sparse_values=0.0,
                    default_value=1.0)
                cell_outputs = tf.add(tf.multiply(
                    cell_outputs, blacklist), minimum_activation * (1 - blacklist))

            beam_search_output, beam_search_state = beam_search_decoder._beam_search_step(
                time=time,
                logits=cell_outputs,
                next_cell_state=next_cell_state,
                beam_state=state,
                batch_size=batch_size,
                beam_width=beam_width,
                end_token=end_token,
                length_penalty_weight=length_penalty_weight,
                coverage_penalty_weight=coverage_penalty_weight)

            finished = beam_search_state.finished
            sample_ids = beam_search_output.predicted_ids
            next_inputs = control_flow_ops.cond(
                math_ops.reduce_all(finished), lambda: self._start_inputs,
                lambda: self._embedding_fn(sample_ids))

        return (beam_search_output, beam_search_state, next_inputs, finished)


def _length_penalty(sequence_lengths, penalty_factor):
    """Calculates the length penalty. See https://arxiv.org/abs/1609.08144.

    Returns the length penalty tensor:
    ```
    [(5+sequence_lengths)/6]**penalty_factor
    ```
    where all operations are performed element-wise.

    Args:
            sequence_lengths: `Tensor`, the sequence lengths of each hypotheses.
            penalty_factor: A scalar that weights the length penalty.

    Returns:
            If the penalty is `0`, returns the scalar `1.0`.  Otherwise returns
            the length penalty factor, a tensor with the same shape as
            `sequence_lengths`.
    """
    penalty_factor = ops.convert_to_tensor(
        penalty_factor, name="penalty_factor")
    penalty_factor.set_shape(())  # penalty should be a scalar.
    static_penalty = tensor_util.constant_value(penalty_factor)
    if static_penalty is not None and static_penalty == 0:
        return 1.0
    return math_ops.div((5. + math_ops.to_float(sequence_lengths))
                        ** penalty_factor, (5. + 1.)**penalty_factor)


def _mask_probs(probs, eos_token, finished):
    """Masks log probabilities.

    The result is that finished beams allocate all probability mass to eos and
    unfinished beams remain unchanged.

    Args:
            probs: Log probabilities of shape `[batch_size, beam_width, vocab_size]`
            eos_token: An int32 id corresponding to the EOS token to allocate
                    probability to.
            finished: A boolean tensor of shape `[batch_size, beam_width]` that
                    specifies which elements in the beam are finished already.

    Returns:
            A tensor of shape `[batch_size, beam_width, vocab_size]`, where unfinished
            beams stay unchanged and finished beams are replaced with a tensor with all
            probability on the EOS token.
    """
    vocab_size = array_ops.shape(probs)[2]
    # All finished examples are replaced with a vector that has all
    # probability on EOS

    # Set the prob mass of the entire beam to EOS
    # this will ensure that the hypotheses that were completed will not turn up again and thus not be considered for TopK
    finished_row = tf.ones([vocab_size], probs.dtype) * probs.dtype.min
    finished_probs = array_ops.tile(
        array_ops.reshape(finished_row, [1, 1, -1]),
        array_ops.concat([array_ops.shape(finished), [1]], 0))
    finished_mask = array_ops.tile(
        array_ops.expand_dims(finished, 2), [1, 1, vocab_size])

    return array_ops.where(finished_mask, finished_probs, probs)

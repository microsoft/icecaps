from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from tensorflow.contrib.seq2seq.python.ops.beam_search_decoder import BeamSearchDecoderState, BeamSearchDecoderOutput


_transpose_batch_time = rnn._transpose_batch_time  # pylint: disable=protected-access
_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access


def _create_zero_outputs(size, dtype, batch_size):
    """Create a zero outputs Tensor structure."""

    def _create(s, d):
        return _zero_state_tensors(s, batch_size, d)

    return nest.map_structure(_create, size, dtype)


def dynamic_decode(decoder,
                   output_time_major=False,
                   impute_finished=False,
                   maximum_iterations=None,
                   parallel_iterations=32,
                   swap_memory=False,
                   scope=None,
                   repetition=None):
    """Perform dynamic decoding (length-wise) with `decoder`.

    Calls initialize() once and step() repeatedly on the Decoder object.

    Args:
      decoder: A `Decoder` instance.
      output_time_major: Python boolean.  Default: `False` (batch major).  If
        `True`, outputs are returned as time major tensors (this mode is faster).
        Otherwise, outputs are returned as batch major tensors (this adds extra
        time to the computation).
      impute_finished: Python boolean.  If `True`, then states for batch
        entries which are marked as finished get copied through and the
        corresponding outputs get zeroed out.  This causes some slowdown at
        each time step, but ensures that the final state and outputs have
        the correct values and that backprop ignores time steps that were
        marked as finished.
      maximum_iterations: `int32` scalar, maximum allowed number of decoding
         steps.  Default is `None` (decode until the decoder is fully done).
      parallel_iterations: Argument passed to `tf.while_loop`.
      swap_memory: Argument passed to `tf.while_loop`.
      scope: Optional variable scope to use.
      repetition: Apply repetition penalty while running beam-search?
        Default is `None` (no repetition penalty)

    Returns:
      `(final_hypotheses, final_input_ids, final_scores)`.

    Raises:
      TypeError: if `decoder` is not an instance of `Decoder`.
      ValueError: if `maximum_iterations` is provided but is not a scalar.
    """
    def _create_hypotheses_ta(s, d):
        return tensor_array_ops.TensorArray(
            dtype=d,
            size=0,
            dynamic_size=True,
            element_shape=s)

    init_ha = _create_hypotheses_ta(
        [maximum_iterations + (not decoder._use_go_tokens), ], tf.int32)
    init_ia = _create_hypotheses_ta([1, ], tf.int64)
    init_sa = _create_hypotheses_ta([1, ], tf.float32)

    def _is_xla_tensor(tensor):
        try:
            op = tensor.op
        except AttributeError:
            return False
        if control_flow_util.IsInXLAContext(op):
            return True
        return False

    with variable_scope.variable_scope(scope, "decoder") as varscope:
        # Properly cache variable values inside the while_loop
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        if maximum_iterations is not None:
            maximum_iterations = ops.convert_to_tensor(
                maximum_iterations, dtype=dtypes.int32, name="maximum_iterations")
            if maximum_iterations.get_shape().ndims != 0:
                raise ValueError("maximum_iterations must be a scalar")
        initial_finished, initial_inputs, initial_state = decoder.initialize()

        zero_outputs = _create_zero_outputs(decoder.output_size,
                                            decoder.output_dtype,
                                            decoder.batch_size)

        is_xla = False
        if any([_is_xla_tensor(i) for i in nest.flatten(initial_inputs)]):
            is_xla = True
        if is_xla and maximum_iterations is None:
            raise ValueError(
                "maximum_iterations is required for XLA compilation.")
        if maximum_iterations is not None:
            initial_finished = math_ops.logical_or(
                initial_finished, 0 >= maximum_iterations)
        initial_sequence_lengths = array_ops.zeros_like(
            initial_finished, dtype=dtypes.int32)
        initial_time = constant_op.constant(0, dtype=dtypes.int32)

        def _shape(batch_size, from_shape):
            if (not isinstance(from_shape, tensor_shape.TensorShape) or
                    from_shape.ndims == 0):
                return tensor_shape.TensorShape(None)
            else:
                batch_size = tensor_util.constant_value(
                    ops.convert_to_tensor(
                        batch_size, name="batch_size"))
                return tensor_shape.TensorShape([batch_size]).concatenate(from_shape)

        dynamic_size = maximum_iterations is None or not is_xla

        def _create_ta(s, d):
            return tensor_array_ops.TensorArray(
                dtype=d,
                size=0 if dynamic_size else maximum_iterations,
                dynamic_size=dynamic_size,
                element_shape=_shape(decoder.batch_size, s), clear_after_read=False)

        initial_outputs_ta = nest.map_structure(_create_ta, decoder.output_size,
                                                decoder.output_dtype)

        if not decoder._use_go_tokens:
            first_output = BeamSearchDecoderOutput(
                scores=decoder._start_token_logits,
                predicted_ids=decoder._start_tokens,
                parent_ids=decoder._start_tokens,)
            initial_outputs_ta = nest.map_structure(lambda ta, out: ta.write(0, out),
                                                    initial_outputs_ta, first_output)

        def condition(unused_time, unused_outputs_ta, unused_state, unused_inputs,
                      finished, unused_sequence_lengths, h_ta, i_ta, s_ta, base_index):
            return math_ops.logical_not(math_ops.reduce_all(finished))

        def body(time, outputs_ta, state, inputs, finished, sequence_lengths, hypotheses, input_ids, scores, base_index):
            """Internal while_loop body.

            Args:
              time: scalar int32 tensor.
              outputs_ta: structure of TensorArray.
              state: (structure of) state tensors and TensorArrays.
              inputs: (structure of) input tensors.
              finished: bool tensor (keeping track of what's finished).
              sequence_lengths: int32 tensor (keeping track of time of finish).
              hypotheses: structure of TensorArray (stores hypotheses so far).
              input_ids: structure of TensorArray.
              scores: structure of TensorArray.
              base_index:  int32 tensor (keeping track of size of the above 3 TensorArrays)

            Returns:
              `(time + 1, outputs_ta, next_state, next_inputs, next_finished,
                next_sequence_lengths, new_hypotheses, new_input_ids, new_scores, new_base)`.
              ```
            """
            (next_outputs, decoder_state, next_inputs,
             decoder_finished) = decoder.step(time, inputs, state)
            if decoder.tracks_own_finished:
                next_finished = decoder_finished
            else:
                next_finished = math_ops.logical_or(decoder_finished, finished)
            next_sequence_lengths = array_ops.where(
                math_ops.logical_not(next_finished),
                array_ops.fill(array_ops.shape(sequence_lengths),
                               time + 1 + (not decoder._use_go_tokens)),
                sequence_lengths)

            nest.assert_same_structure(state, decoder_state)
            nest.assert_same_structure(outputs_ta, next_outputs)
            nest.assert_same_structure(inputs, next_inputs)

            # Zero out output values past finish
            if impute_finished:
                emit = nest.map_structure(
                    lambda out, zero: array_ops.where(
                        next_finished, zero, out),
                    next_outputs,
                    zero_outputs)
            else:
                emit = next_outputs

            # Copy through states past finish
            def _maybe_copy_state(new, cur):
                # TensorArrays and scalar states get passed through.
                if isinstance(cur, tensor_array_ops.TensorArray):
                    pass_through = True
                else:
                    new.set_shape(cur.shape)
                    pass_through = (new.shape.ndims == 0)
                return new if pass_through else array_ops.where(finished, cur, new)

            outputs_ta = nest.map_structure(lambda ta, out: ta.write(time + (not decoder._use_go_tokens), out),
                                            outputs_ta, emit)

            # Extract hypotheses, scores for reference
            outputs_so_far = nest.map_structure(
                lambda ta: ta.stack(), outputs_ta)
            parent_ids = outputs_so_far.parent_ids
            hypotheses_so_far = outputs_so_far.predicted_ids
            forward_scores = next_outputs.scores

            sl = tf.ones([decoder.batch_size], tf.int32) * \
                (time + 1 + (not decoder._use_go_tokens))
            hypotheses_so_far = beam_search_ops.gather_tree(
                hypotheses_so_far,
                parent_ids,
                max_sequence_lengths=sl,
                end_token=decoder._end_token)

            # Add repetition penalty
            if repetition != 0:
                def unique_counter(x): return tf.cast(
                    tf.size(tf.unique(x)[0]), tf.float32)

                def wrapper_rep_penalty_function(sentence):
                    def first_time_penalty():
                        total_unique_words = unique_counter(sentence)
                        return tf.log(total_unique_words)

                    def generic_penalty():
                        sentence_length = tf.shape(sentence)[0]
                        total_unique_words = unique_counter(sentence)
                        unique_words_before = unique_counter(
                            sentence[:sentence_length - 1])
                        return (tf.log(total_unique_words) - tf.log(unique_words_before))

                    return repetition * tf.cond(math_ops.equal(tf.shape(sentence)[0], 1),
                                                true_fn=first_time_penalty,
                                                false_fn=generic_penalty)

                # Use reshaped hypotheses; calculate penalty per beam per batch
                transposed_hypotheses = tf.transpose(
                    hypotheses_so_far, [2, 1, 0])
                repetition_penalty = tf.map_fn(lambda x: tf.map_fn(
                    wrapper_rep_penalty_function, x, dtype=tf.float32), transposed_hypotheses, dtype=tf.float32)
                repetition_penalty = tf.transpose(repetition_penalty, [1, 0])
                forward_scores += repetition_penalty

                # Add repetition penalty hypothesis scores
                decoder_state = BeamSearchDecoderState(
                    cell_state=decoder_state.cell_state,
                    log_probs=decoder_state.log_probs + repetition_penalty,
                    finished=decoder_state.finished,
                    lengths=decoder_state.lengths,
                    accumulated_attention_probs=decoder_state.accumulated_attention_probs)

            if impute_finished:
                next_state = nest.map_structure(
                    _maybe_copy_state, decoder_state, state)
            else:
                next_state = decoder_state

            finished_this_beam = math_ops.logical_and(
                math_ops.logical_not(finished), decoder_finished)
            # Make sure number of outputs is never zero
            finished_this_beam = tf.cond(math_ops.logical_and(math_ops.logical_and(tf.equal(hypotheses.size(), 0),
                                                                                   tf.equal(tf.size(tf.where(finished_this_beam)), 0)),
                                                              tf.equal(time, maximum_iterations - 1)),
                                         true_fn=lambda: tf.cast(tf.ones_like(
                                             finished_this_beam), dtype=tf.bool),
                                         false_fn=lambda: finished_this_beam)

            def prepare_hypotheses_for_ta():
                finished_beams = tf.where(finished_this_beam)

                hypotheses_for_ta = tf.boolean_mask(tf.transpose(
                    hypotheses_so_far, [1, 2, 0]), finished_this_beam)

                # Pad hypotheses with EOS token
                hypotheses_for_ta = tf.pad(hypotheses_for_ta,
                                           [[0, 0], [
                                               0, maximum_iterations + (not decoder._use_go_tokens) - tf.shape(hypotheses_for_ta)[-1]]],
                                           constant_values=decoder._end_token)

                input_query_id = tf.expand_dims(finished_beams[:, 0], 1)
                scores_forward = tf.expand_dims(tf.boolean_mask(
                    forward_scores, finished_this_beam), 1)

                def inner_cond(index, base, hyp_ta, ind_ta, score_ta, hypos, input_ids, forward_scores):
                    # Populate TA with given elements AND do not consider blank responses
                    return math_ops.logical_and(math_ops.less(index, tf.shape(hypos)[0]), math_ops.greater(time, 0 - (not decoder._use_go_tokens)))

                def inner_body(index, base, hyp_ta, ind_ta, score_ta, hypos, input_ids, forward_scores):
                    new_hyp_ta = nest.map_structure(lambda ta, out: ta.write(base, out),
                                                    hyp_ta, hypos[index])
                    new_ind_ta = nest.map_structure(lambda ta, out: ta.write(base, out),
                                                    ind_ta, input_ids[index])

                    # Remove repetition penalty from stored score, use as a feature for later re-reranking
                    forward_scores_store = forward_scores[index]
                    if repetition != 0:
                        forward_scores_store -= repetition * \
                            unique_counter(hypos[index])

                    # Normalize finished scores by their length
                    new_scores_ta = nest.map_structure(lambda ta, out: ta.write(base, out),
                                                       score_ta, forward_scores_store / tf.cast(tf.count_nonzero(hypos[index] - decoder._end_token), tf.float32))

                    return (index + 1, base + 1, new_hyp_ta, new_ind_ta, new_scores_ta, hypos, input_ids, forward_scores)

                # Add multiple hypotheses (and related information) to TensorArray using a while_loop
                inner_result = tf.while_loop(
                    inner_cond,
                    inner_body,
                    loop_vars=(
                        tf.constant(0),
                        base_index,
                        hypotheses,
                        input_ids,
                        scores,
                        hypotheses_for_ta,
                        input_query_id,
                        scores_forward
                    ),
                    parallel_iterations=parallel_iterations,
                    swap_memory=swap_memory)
                return inner_result[1], inner_result[2], inner_result[3], inner_result[4]

            # In case finished is not True for any beams
            new_base, new_hypotheses, new_input_ids, new_scores = tf.cond(
                math_ops.greater(tf.count_nonzero(finished_this_beam), 0),
                true_fn=prepare_hypotheses_for_ta,
                false_fn=lambda: (base_index, hypotheses, input_ids, scores))

            return (time + 1, outputs_ta, next_state, next_inputs, next_finished,
                    next_sequence_lengths, new_hypotheses, new_input_ids, new_scores, new_base)

        res = control_flow_ops.while_loop(
            condition,
            body,
            loop_vars=(
                initial_time,
                initial_outputs_ta,
                initial_state,
                initial_inputs,
                initial_finished,
                initial_sequence_lengths,
                init_ha,
                init_ia,
                init_sa,
                tf.constant(0),
            ),
            parallel_iterations=parallel_iterations,
            maximum_iterations=maximum_iterations,
            swap_memory=swap_memory)

        final_hypotheses_ta = res[6]
        final_input_ids_ta = res[7]
        final_scores_ta = res[8]

        final_hypotheses = nest.map_structure(
            lambda ta: ta.stack(), final_hypotheses_ta)
        final_input_ids = nest.map_structure(
            lambda ta: ta.stack(), final_input_ids_ta)
        final_scores = nest.map_structure(
            lambda ta: ta.stack(), final_scores_ta)

    return final_hypotheses, final_input_ids, final_scores

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import basic_decoder
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import nest


class BasicDecoderOutput(
        collections.namedtuple("BasicDecoderOutput", ("scores", "rnn_output", "sample_id"))):
    pass


class BasicDecoder(basic_decoder.BasicDecoder):
    """Basic sampling decoder (for MMI)."""

    #'''
    @property
    def output_size(self):
        # Return the cell output and the id
        return BasicDecoderOutput(
            scores=self._rnn_output_size(),
            rnn_output=self._rnn_output_size(),
            sample_id=self._helper.sample_ids_shape)

    @property
    def output_dtype(self):
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and the sample_ids_dtype from the helper.
        dtype = nest.flatten(self._initial_state)[0].dtype
        return BasicDecoderOutput(
            nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            self._helper.sample_ids_dtype)

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
        with ops.name_scope(name, "BasicCustomDecoderStep", (time, inputs, state)):
            cell_outputs, cell_state = self._cell(inputs, state)
            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)

            # Calculate probabilities at each step
            step_log_probs = nn_ops.log_softmax(cell_outputs)

            sample_ids = self._helper.sample(
                time=time, outputs=cell_outputs, state=cell_state)
            (finished, next_inputs, next_state) = self._helper.next_inputs(
                time=time,
                outputs=cell_outputs,
                state=cell_state,
                sample_ids=sample_ids)
        outputs = BasicDecoderOutput(step_log_probs, cell_outputs, sample_ids)
        return (outputs, next_state, next_inputs, finished)
        
    '''

    @property
    def output_size(self):
        # Return the cell output and the id
        return BasicDecoderOutput(

            rnn_output=self._rnn_output_size(),
            sample_id=self._helper.sample_ids_shape)

    @property
    def output_dtype(self):
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and the sample_ids_dtype from the helper.
        dtype = nest.flatten(self._initial_state)[0].dtype
        return BasicDecoderOutput(

            nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            self._helper.sample_ids_dtype)


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
      with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
        cell_outputs, cell_state = self._cell(inputs, state)
        if self._output_layer is not None:
          cell_outputs = self._output_layer(cell_outputs)
        sample_ids = self._helper.sample(
            time=time, outputs=cell_outputs, state=cell_state)
        (finished, next_inputs, next_state) = self._helper.next_inputs(
            time=time,
            outputs=cell_outputs,
            state=cell_state,
            sample_ids=sample_ids)
      outputs = BasicDecoderOutput(tf.zeros(tf.shape(sample_ids)[0]), cell_outputs, sample_ids)
      return (outputs, next_state, next_inputs, finished)
      '''
"""

The module containing the ACT controller

"""

import trax
import textwrap

from typing import Optional, Dict, List, Tuple, Any, Union
from trax import fastmath
from trax.fastmath import numpy as np


class ACTController:
  """
  Adaptive Computation Time (ACT) Controller for dynamic computation processes.

  ACT (Adaptive Computation Time) is a technique used primarily in neural networks
  to dynamically adjust the number of computation steps based on the complexity of the input data.
  Key concepts of ACT include:
  - Formation of a halting probability each iteration, which determines when to stop iterating.
    Iteration halts when the cumulative halting probability is about to exceed one.
  - The output is a weighted sum (or superposition) of the states at each iteration, weighted by
    their respective halting probabilities. This superposition allows for a fine-grained balance
    between computation at each step.
  - The superposition nature of the output makes it compatible with gradient descent, allowing
    backpropagation to effectively adjust model parameters.

  This class provides a flexible implementation of ACT by managing multiple data streams or 'channels'
  and allowing for dynamic halting based on cumulative halting probabilities.

  Channels:

  Manages multiple 'channels' of data, each representing a separate stream
  of data accumulation. Channels can handle different aspects of the computation
  process, such as states, outputs, or intermediate calculations, and accumulate
  data independently across iterations.

  Key Features:
  - Handles multiple channels for parallel data management.
  - Manages nested data structures including dictionaries, lists, and tuples.
  - Tracks halting statistics to monitor the number of iterations required for halting.
  - Supports batch dimensions for parallel processing of multiple data sets.
  - Provides comprehensive error handling with detailed messages for troubleshooting.
  - Lazily infers batch shape from the first tensor data, eliminating the need for users
    to specify the exact shape upfront, only the number of batch dimensions.

  Configuration Options:
  - `num_batch_dims` (int): Specify the number of batch dimensions in tensors.
  - `channels` (Union[List[str], Dict[str, Optional[Any]]]):
      * List of channel names: Initializes empty channels to be filled during the process.
      * Dictionary with channel names and initial values: Pre-configures channels with given tensors.
  - `cum_halting_probability` (Optional[np.ndarray]): Initial tensor for cumulative halting probabilities.
      * Generally, you would use this to pick up on a halted act process, or ensure your halting probabilities
        are already partially exhausted.
  - `epsilon` (float): Small constant used for halting threshold.

  Usage:
  1. Initialize the class with batch dimensions and channel names.
  2. Buffer tensors into channels using `buffer_tensors`.
  3. Call `update_channels` with halting probabilities to process and accumulate data.
  4. Access accumulated data using `get_channel` or channel indexing.

  Code Example 1 - Basic Usage:*
  ```python
  act_controller = ACTController(num_batch_dims=1, channels=["state", "output"])
  while not act_controller.is_halted:
      state_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
      output_tensor = np.array([[0.5, 0.5], [0.6, 0.4]])
      act_controller.buffer_tensors("state", state_tensor)
      act_controller.buffer_tensors("output", output_tensor)
      halting_probabilities = np.array([0.5, 0.5])
      act_controller.update_channels(halting_probabilities)
  accumulated_state = act_controller.get_channel("state")
  accumulated_output = act_controller.get_channel("output")
  ```

  Code Example 2 - Accumulating Nested Data:
  ```python
  act_controller = ACTController(num_batch_dims=1, channels=["composite_data"])
  nested_data = {"vector": np.array([1.0, 2.0]), "matrix": np.array([[1, 2], [3, 4]])}
  while not act_controller.is_halted:
      act_controller.buffer_tensors("composite_data", nested_data)
      halting_probabilities = np.array([0.7])
      act_controller.update_channels(halting_probabilities)
  accumulated_data = act_controller.get_channel("composite_data")
  ```

  Properties:
  - `num_batch_dims` (int): Number of batch dimensions in tensors.
  - `batch_shape` (Tuple[int, ...]): Shape of the batch dimensions.
  - `batches_halted` (np.ndarray): Indicates if each batch element has halted.
  - `channel_names` (Tuple[str, ...]): Names of the managed channels.
  - `channels` (Dict[str, Optional[Any]]): Data accumulators for each channel.
  - `residuals` (Optional[np.ndarray]): Residuals for halting.
  - `num_halting_steps` (Optional[np.ndarray]): Statistics of halting iterations.

  Public Methods:
  - `buffer_tensors`: Buffers tensors into a specified channel.
  - `update_channels`: Updates channels with buffered tensors based on halting probabilities.
  - `get_channel`: Retrieves data for a specified channel.
  - `__getitem__`: Allows channel data access using indexing.
  """

  def _validate_structures_same(self,
                                structures: Tuple[Any, Any],
                                names: Tuple[str, str],
                                error_activitity: str
                                ):
    # There are several places where we could have nested structures
    # that must be the same between act iterations. This internal
    # function is designed to walk through such structures and validate the
    # shapes are the same

    def check_shapes_same(structure_one: np.ndarray,
                          structure_two: np.ndarray)->None:
      if structure_one.shape != structure_two.shape:
        msg =f"""
        In the leafs of the tensor structures, the
        structure of name {names[0]} had shape {structure_one.shape}.
        However, {names[1]} had shape {structure_two.shape}. This
        is not allowed
        """
        msg = textwrap.dedent(msg)
        raise ValueError(msg)
      return None
    fastmath.nested_map_multiarg(check_shapes_same, *structures, ignore_nones=True)

  def _validate_halting_probabilities(self, halting_probabilities: np.ndarray)->None:
    if np.any(halting_probabilities > 1) or np.any(halting_probabilities < 0):
      msg = """
      Attempting to update the channels using the halting probabilities. An
      issue was encountered

      The halting probabilities were expected to be between zero and one,
      but were outside this region.

      Possible Resolutions: Check that you are providing the right tensor,
      and activating it correctly.
      """
      msg = textwrap.dedent(msg)
      raise ValueError(msg)

    if halting_probabilities.shape != self.batch_shape:
      msg = f"""
      Attempting to update the channels using the halting probabilitie. An
      issue was encountered

      The batch shape is {self.batch_shape} while the shape of the
      halting probabilities is {halting_probabilities.shape}.

      These do not match, but they should.

      Possible Resolution: Check that you have set the number of batch
      dimensions correctly in the constructor, and if you are manually setting
      up channels, that you are feeding the right tensors in
      """
      msg = textwrap.dedent(msg)
      raise ValueError(msg)


  def _validate_buffer_name(self, name: str):
    if name not in self.channel_names:
      msg = f"""
      Attempting to buffer to channel '{name}', but this channel was never setup.

      Possible Resolution: Either check your spelling, or provide another channel called {name} in the constructor"""
      msg = textwrap.dedent(msg)
      raise KeyError(msg)

  def _validate_buffer_set(self,
                           name: str,
                           value: np.ndarray
                           ):
    """
    Validates that a buffer being set to by name
    with value can actually be done. This is
    to set to an unfilled buffer. Reset is handled
    elsewhere.
    """

    if self._get_buffer(name) is not None:
      msg = f"""
      Attempting to buffer to channel '{name}', but '{name}' has already
      been buffered this iteration.

      Possible Resolution: Check your code, and figure out why you are buffering to the same key twice
      per act iteration. You may only buffer once per channel per iteration
      """
      msg = textwrap.dedent(msg)
      raise RuntimeError(msg)

    if self.is_halted:
      msg = f"""
      Attempting to buffer to channel '{name}', but act process
      has already halted and should not be further processed.

      Possible Resolution: Remember to put while not your_instance.is_halted:
      as your driving loop, so your code stops looping when halted
      """
      msg = textwrap.dedent(msg)
      raise RuntimeError(msg)
    # This deserves some commentary, as it concerns the
    # lazy initialization logic
    #
    # When setting a channel that has already setup,
    # we demand that the buffer and channel shape are the same.
    #
    # If it has not been setup,

    if self.get_channel(name, True) is not None:
      # Shapes must be the same between ACT rounds
      structures = (self.get_channel(name, True), value)
      names = ("inferred channel", "provided buffer")
      try:
        self._validate_structures_same(structures, names, f"buffering tensor named '{name}'")
      except ValueError as err:
        msg = f"""
          Attempting to buffer to channel '{name}', but buffer and channel
          shape were not the same.
    
          Possible Resolution: Make sure your tensors are the same shape between
          act iterations
          """
        msg = textwrap.dedent(msg)
        raise ValueError(msg) from err
    else:
      # This channel was never setup. Set it up by
      # walking the tree and making zeros like, then
      # storing the update.
      self._manage_batch_shape(value, f"buffering tensor named %s" % name)
      empty_accumulator = fastmath.nested_map(lambda x: np.zeros_like(x),
                                              value,
                                              ignore_nones=True)
      self._set_channel(name, empty_accumulator)


  def _expand_dims_to_len(self, tensor: np.ndarray, length: int)->np.ndarray:
    # Unsqueezes last dimensions to ensure batch dimensions match,
    # and broadcasting works on later dimensions.
    #
    # Accepts the tensor, and the length to unsqueeze to
    # Returns the tensor.
    while len(tensor.shape) < length:
      tensor = np.expand_dims(tensor, -1)
    return tensor

  def _gather_nested_shapes(self, structure: Any)->List[Any]:
    # Gathers shape information off a nested structure
    # and returns them in a flat list.

    # This is deterministic
    output = []
    def gather_shapes(x):
      output.append(x.shape)
    trax.fastmath.nested_map(gather_shapes, structure, ignore_nones=True)
    return output

  def _lazy_initialize_tensors(self,
                               shape: Tuple[int,...],
                               dtype: np.dtype = None,
                               ):
    # An internal helper function, this will
    # perform all required setup actions once the
    # batch shape is determined
    if self._get_probs() is None:
      accumulator = np.zeros(shape, dtype)
      self._set_probs(accumulator, "Initializing probability accumulator")
    if self.get_residuals(True) is None:
      accumulator = np.zeros(shape, dtype)
      self._set_residuals(accumulator)
    if self._num_halting_steps is None:
      accumulator = np.zeros(shape, dtype=np.int32)
      self._set_num_halting_steps(accumulator)

  def _manage_batch_shape(self,
                          structure: Any,
                          error_task_source: str
                          ):

    # An internal helper function which both manages and validates the
    # batch shape
    #
    # This may be passed a tensor or nested tensor structure, and
    # will travel through the structure or tensor checking shapes.
    # To make good error messages, the logic also involves passing
    # in an indicator that tells what was being done.
    #
    # Should a batch shape not yet be set, the first shape seen is grabbed
    # and set as the batch shape, if available. Otherwise, we enforce
    # that all batch shapes are the same, and raise a ValueError if not.

    leaves = fastmath.tree_leaves(structure)
    for leaf in leaves:
      batch_shape = leaf.shape[:self.num_batch_dims]
      if self.batch_shape is None:
        # Store expected shape, and error information
        self._batch_info["shape"] = batch_shape
        self._batch_info["source"] = error_task_source

        # Setup the probability accumulator, if needed
        self._lazy_initialize_tensors(batch_shape, leaf.dtype)

      elif self.batch_shape != batch_shape:
        # Raise error
        msg = f""" Inferred and provided batch shape not equal.
              Expected {self.batch_shape} based on {self._batch_info["source"]},
              got {batch_shape} from {error_task_source}"""
        msg = textwrap.dedent(msg)
        raise ValueError(msg)

  @property
  def is_halted(self)->bool:
    if self.cum_halting_prob is None:
      return False
    return np.all(self.batches_halted)

  @property
  def batch_shape(self)->Tuple[int,...]:
    return self._batch_info["shape"]

  @property
  def batches_halted(self)->np.ndarray:
    return self._get_probs() > 1 - self.epsilon

  @property
  def channel_names(self)->Tuple[str, ...]:
    return tuple(self._channels.keys())

  @property
  def channels(self)->Dict[str, Optional[Any]]:
    return {name : self.get_channel(name, True) for name in self.channel_names}

  @property
  def buffers(self)->Dict[str, Optional[Any]]:
    return {name : self._get_buffer(name) for name in self.channel_names}

  @property
  def residuals(self)->Optional[np.ndarray]:
    return self.get_residuals()

  @property
  def num_halting_steps(self)->Optional[np.ndarray]:
    return self._get_num_halting_steps()

  @property
  def cumulative_halting_probabilities(self)->Optional[np.ndarray]:
    return self._get_probs()

  @property
  def cum_halting_prob(self)->Optional[np.ndarray]:
    return self._halting_prob

  # Internally, buffers and channels are interacted with
  # only using manually defined getters and setters.
  #
  # This is implemented that way so batch shrinking is easy
  # to implement in the future if needed

  def _get_probs(self):
    return self._halting_prob

  def _set_probs(self, value, error_task: str):
    self._manage_batch_shape(value, error_task)
    self._halting_prob = value


  def _get_buffer(self,
                  name: str)->np.ndarray:
    assert name in self._buffers, "Internal error in _get_buffer: Contact maintainer"
    return self._buffers[name]

  def _set_buffer(self,
                  name: str,
                  value: np.ndarray)->None:

    """
    The setter for the buffer.

    Contains most of the validation logic, alongside
    the final set mechanism. We also have some lazy
    initialization logic in here.
    """

    # Peform validation then set the buffer.
    self._validate_buffer_name(name)
    self._validate_buffer_set(name, value)
    self._buffers[name] = value

  def _reset_buffer(self,
                    name: str)->None:
    """
    Resetter for a buffer.
    """
    self._validate_buffer_name(name)
    self._buffers[name] = None


  def _set_channel(self,
                   name: str,
                   value: Any)->None:
    self._manage_batch_shape(value, "setting channel %s" % name)
    self._channels[name] = value

  def _set_num_halting_steps(self, value: np.ndarray)->None:
    self._num_halting_steps = value

  def _get_num_halting_steps(self)->np.ndarray:
    return self._num_halting_steps

  def _set_residuals(self, value)->None:
    self._residuals = value

  def get_residuals(self, force: bool = False):
    """
    Get the residuals for this iteration. Will throw if
    not forcing and not halting
    """
    if not self.is_halted and not force:
      msg = f"""
      Issue when retrieving residuals.

      Attempted to retrieve residuals when act process
      has not halted, and without forcing

      Potential Resolution: Check your code and make
      sure the act process is halting properly, or
      use .get_residuals(force=True)
      """
      msg = textwrap.dedent(msg)
      raise RuntimeError(msg)
    return self._residuals



  def get_channel(self,
                  name: str,
                  force: bool = False
                  )->Optional[Any]:
    """
    Get the value associated with a channel, if the
    halting state has been reached. Can be forced to
    retrieve even if not halted.
    """
    if not self.is_halted and not force:
      msg = f"""
      Issue when retrieving a channel. Act process
      has not yet halted. Occurred when retrieving
      channel of name {name}

      Possible Resolution: Check your logic and verify you
      are properly halting. Or, use .get_channel(name, force=True)
      """
      msg = textwrap.dedent(msg)
      raise RuntimeError(msg)
    return self._channels[name]

  def __getitem__(self, name: str):
    return self.get_channel(name)

  def _compute_halting_masks(self,
                             halting_probabilities: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes, from the halting probabilities, the halting masks. These
    are bool tensors that tell us information on act halt status
    :param halting_probabilities:
    :return:
      is_halted: np.ndarray, tells us if before the halting update the current iteration is halted
      will_be_halted: np.ndarray, tells us after the halting update what is halted
      will_be_newly_halted: np.ndarray, tells us what is halting this particular update.
    """
    is_halted = self.batches_halted
    will_be_halted = self.cum_halting_prob + halting_probabilities > 1 - self.epsilon
    will_be_newly_halted = will_be_halted & ~is_halted
    return is_halted, will_be_halted, will_be_newly_halted

  def _manage_residuals(self,
                        halting_probabilities: np.ndarray,
                        will_be_newly_halted: np.ndarray,
                        will_be_halted: np.ndarray
                        )->np.ndarray:
    """
    Manage the residual based probability actions. Residuals are used
    to ensure that probabilities consistently add up to one, by replacing
    halting probabilities that would be too high with a clamped value.

    This includes storing away the residuals when halting is reached,
    and clamping the halting probabilities so the cumulative probabilities add up
    to one.

    Some models use what residual was used when clamping for normalization. We
    also store the residual.

    :param halting_probabilities: The unclamped halting probs
    :param will_be_newly_halted: The batch elements that are newly halting
    :param will_be_halted: The batch elements to perform residual replacement on
    :return: The clamped halting probabilities
    """

    residuals = 1 - halting_probabilities
    self._set_residuals(np.where(will_be_newly_halted, residuals, self.get_residuals(True)))
    halting_probabilities = np.where(will_be_halted, residuals, halting_probabilities)
    return halting_probabilities

  def _manage_statistics(self, is_halted: np.ndarray)->None:
    """
    A catch_all location to record metric quantities useful
    for evaluating the act process.
    """

    # At the moment there is only one statistic being gathered, in
    # addition to residuals, and that is the num_halting_steps metric

    num_halting_steps = self._get_num_halting_steps() + ~is_halted
    self._set_num_halting_steps(num_halting_steps)


  def _manage_cumulative_probabilities(self,
                                       halting_probabilities: np.ndarray
                                       )->Tuple[np.ndarray, np.ndarray]:
    """
    Manage the halting probability effectively and efficently.

    This includes storing away residuals and statistical
    quantities, and any similar actions.

    returns:
      a boolean batchlike mask. True will indicate this region
      was originally halted and should not be updated. False means update

      a updated halting probabilities tensor.
    """

    self._validate_halting_probabilities(halting_probabilities)

    # Compute and store residuals. Use residuals to update halting probabilities
    # requiring clamping. Cumulative probabilities must add up to one, so when
    # we would exceed one we set the halting probabilities directly to something
    # that adds up to one. This value is also stored as a residual.

    is_halted, will_be_halted, will_be_newly_halted = self._compute_halting_masks(halting_probabilities)
    halting_probabilities = self._manage_residuals(halting_probabilities,
                                                   will_be_newly_halted,
                                                   will_be_halted)


    # Update statistics and probabilities.

    self._manage_statistics(is_halted)
    update = self._get_probs() + halting_probabilities
    update = np.where(is_halted, self._get_probs(), update)
    self._set_probs(update, "updating_probabilities")
    return is_halted, halting_probabilities

  def buffer_tensors(self,
                     name: str,
                     tensor_structure: Any):
    """
    Buffers a particular defined channel for the ACT process.
    This is called within a particular iteration and will be
    accumulated when halting probabilities are provided. It should
    be provided with the name of the channel to buffer, and the
    accumulation for the iteration.

    Params:
      name: The name of the channel to buffer into.
      tensor_structure: A tensor, or nested tensor structure, to buffer and accumulate
    Raises:
      RuntimeError: When attempting to buffer to an already filled buffer
      RuntimeError: When attempting to buffer after halting state is reached
      KeyError: When buffering to a channel that was never setup
      ValueError: When batch shape does not match during lazy init
      ValueError: When structure shape does not match across ACT steps
    """
    self._set_buffer(name, tensor_structure)

  def update_channels(self,
                      halting_probabilities: np.ndarray):
    """

    A part of the ACT process.

    This method completes an ACT generation round, by
    storing away the buffered elements. On completion
    of this call, the controller may switch from incomplete
    to complete state, or change batches from incomplete to
    complete.

    Parameters:
      halting_probabilities: A np.Array of halting probabilities of batch_shape
    Raises:
      RuntimeError: When a buffer was never set
      RuntimeError: When updating and the controller is already halted
      ValueError: When updating and the halting_probabilities are not probabilities between 0 and 1
      ValueError: When updating and the halting_probabilities are not matching the batch shape
    """

    # Manage halting probabilities. This will include residuals and
    # other misc tasks, and the formation of properly clamped
    # probabilities

    is_halted, halting_probabilities = self._manage_cumulative_probabilities(halting_probabilities)


    # Define helper function
    def make_updated_channel(current_channel, current_buffer):
      # A helper function
      #
      # This expands the mask and halting probabilities to
      # fit the channel dimension, does a weighted update,
      # and keeps only the parts that actually needed to be
      # updated.
      num_dims = len(current_buffer.shape)

      broadcast_halting_probabilities = self._expand_dims_to_len(halting_probabilities, num_dims)
      broadcast_mask = self._expand_dims_to_len(is_halted, num_dims)

      update = current_buffer*broadcast_halting_probabilities
      update = current_channel + update
      update = np.where(broadcast_mask, current_channel, update)
      return update

    # Run accumulate.
    for name in self.channel_names:
      buffer = self._get_buffer(name)
      if buffer is None:
        msg = f"""
        Attempting to update the channels using the halting probabilities. An
        issue was encountered.

        The buffer of name {name} was never set up before attempting to update.

        Possible Resolutions: Ensure your code is consistently setting to this
        channel in each act iteration, or remove it from the configuration in
        the constructor"""
        msg = textwrap.dedent(msg)
        raise RuntimeError(msg)

      # Actual update logic. We expand the halting
      # probability to broadcast correctly, then weight the
      # buffer and add it to the channel.

      update = fastmath.nested_map_multiarg(make_updated_channel,
                                            self.get_channel(name, True),
                                            buffer,
                                            ignore_nones=True)
      self._set_channel(name, update)
      self._reset_buffer(name)


  def __init__(self,
               num_batch_dims: int,
               channels: Union[List[str], Dict[str, Optional[Any]]],
               cum_halting_probability: Optional[Any] = None,
               epsilon: float = 1e-4
               ):

    self.num_batch_dims = num_batch_dims
    self.epsilon = epsilon

    self._num_halting_steps: Optional[np.ndarray] = None
    self._batch_info: Dict[str, Optional[Any]] = {"shape" : None, "source" : None}

    self._channels: Dict[str, Optional[Any]] = {name : None for name in channels}
    self._buffers: Dict[str, Optional[Any]] = {name : None for name in channels}
    self._halting_prob = None
    self._residuals = None

    # Setup and posssible lazy init
    if isinstance(channels, Dict):
      for name, channel in channels.items():
        self._set_channel(name, channel)

    if cum_halting_probability is not None:
      self._set_probs(cum_halting_probability, "initialize: halting_probabilities")




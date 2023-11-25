import unittest
import numpy as np
from numpy import random
from trax_act_controller import ACTController


class TestACTChannels(unittest.TestCase):
  def test_lazy_initialization(self):
    """Test that when setting to a buffer, if the accumulator was not initialized, it is setup with zeros of the correct shape"""
    instance = ACTController(1, ["test"])

    # This should initialize the accumulator too, with a tensor of zeros of the same shape
    instance.buffer_tensors("test", random.standard_normal([5, 10]))

    # Check that actually appeneded
    self.assertTrue(np.all(instance.get_channel("test", force=True) == np.zeros([5,10])))
    self.assertTrue(instance.batch_shape == (5,))

  def test_update(self):
    """Test that update_channels actually can work"""

    update = random.standard_normal([5, 10])
    probs =  random.uniform(0, 1, [5])
    instance = ACTController(1, ["test"])
    instance.buffer_tensors("test", update)
    instance.update_channels(probs)

  def test_complex_update(self):
    """ Test more complex updates work"""

    update_a = random.standard_normal([5, 10])
    update_b = random.standard_normal([5, 10, 20, 3])
    probs =  random.uniform(0, 1, [5, 10])

    instance = ACTController(2, ["test", "test2"])

    instance.buffer_tensors("test", update_a)
    instance.buffer_tensors("test2", update_b)
    instance.update_channels(probs)


  def test_accumulation_logic(self):
    """Test that update actually causes accumulation to occur following the weighted sum"""

    # Setup accumulation values
    setup = np.ones([5, 10])
    update = np.ones([5, 10])
    probs = 0.5*np.ones([5])
    expected = setup + probs[:, None] *update

    # Run test
    instance = ACTController(1, {"test" : setup})
    instance.buffer_tensors("test", update)
    instance.update_channels(probs)

    actual = instance.get_channel("test", force=True)
    self.assertTrue(np.all(expected == actual))


class TestACTChannelsErrors(unittest.TestCase):
  def test_enforces_prob_shape(self):
    """Test that the model enforces the probabilitities to be the batch shapes"""

    # A accumulator which has been setup should
    # expect the probs shape to equal the batch shape.

    setup = np.ones([5, 10])
    update = np.ones([5, 10])
    probs = np.ones([7])

    instance = ACTController(1, {"test" : setup})
    instance.buffer_tensors("test", update)
    with self.assertRaises(ValueError):
      instance.update_channels(probs)

  def test_illegal_access(self):
    """Test that getting a channel, when not halted, throws an error unless forcing"""
    instance = ACTController(1, ["test"])

    with self.assertRaises(RuntimeError):
      instance["test"]

    with self.assertRaises(RuntimeError):
      instance.get_channel("test")

import unittest
import numpy as np
from numpy import random
from trax_act import ACTController

class TestACTBuffer(unittest.TestCase):
  def test_lazy_set(self):
    """ Test that a straightforward lazy buffer works"""
    instance = ACTController(1, ["test"])
    instance.buffer_tensors("test", random.standard_normal([5, 10]))

    # The batch shape should already be known at this point.
    self.assertTrue(instance.batch_shape == (5,))
  def test_configured_set(self):
    """Test that a nonlazy buffer works"""
    instance = ACTController(1, {"test" : random.standard_normal([5, 10])})
    self.assertTrue(instance.batch_shape == (5,))
    instance.buffer_tensors("test", random.standard_normal([5, 10]))

class TestACTBufferErrors(unittest.TestCase):
  def test_buffer_not_setup(self):
    """ Test we throw when setting to a buffer that was not properly setup"""
    example = random.standard_normal([5, 10])
    instance = ACTController(1, ["test"])
    with self.assertRaises(KeyError):
      instance.buffer_tensors("output", example)


  def test_has_halted(self):
    """Test we throw when reaching the halted condition and attempting to continue buffering"""
    example = random.standard_normal([5, 10])
    instance = ACTController(1, {"test" : np.array([1.0])}, np.array([1.0]))
    with self.assertRaises(RuntimeError) as err:
      instance.buffer_tensors("test", example)

  def test_buffer_twice(self):
    """Test that attempting to buffer twice without updating throws"""
    example = random.standard_normal([5, 10])
    instance = ACTController(1, ["test"])
    instance.buffer_tensors("test", example)

    with self.assertRaises(RuntimeError) as err:
      instance.buffer_tensors("test", example)


  def test_never_buffered(self):
    """Test we throw when attempting to advance, but missed buffering an item"""
    example = random.standard_normal([5, 10])
    probs = random.uniform(0, 1, [5])
    instance = ACTController(1, ["test", "not_buffered"], probs)
    instance.buffer_tensors("test", example)
    with self.assertRaises(RuntimeError) as err:
      instance.update_channels(probs)

  def test_unequal_batch_shape(self):
    """ Test we throw when attempting to work with tensors of differing batch dimensions"""

    # This is of concern for the lazy initialization logic.
    instance = ACTController(1, ["test", "test2"])
    instance.buffer_tensors("test", random.standard_normal([10, 20]))
    with self.assertRaises(ValueError) as err:
      instance.buffer_tensors("test2", random.standard_normal([20, 30]))

  def test_bad_shape(self):
    """Test we throw when the shape has been configured, and the buffer does not match"""
    setup =  random.standard_normal([5, 10, 3])
    example = random.standard_normal([5, 10, 5])
    instance = ACTController(1, {"test" : setup})
    with self.assertRaises(ValueError) as err:
      instance.buffer_tensors("test", example)



  def test_bad_broadcastable_shape(self):
    """Test we throw when the shape has been configured, and the buffer does not match"""
    setup = random.standard_normal([5, 10, 3])
    example = random.standard_normal([5, 10, 1])
    instance = ACTController(1, {"test" : setup})
    with self.assertRaises(ValueError) as err:
      instance.buffer_tensors("test", example)


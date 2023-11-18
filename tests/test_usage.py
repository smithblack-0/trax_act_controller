import unittest
import numpy as np
from numpy import random
from src.act import ACTController


class TestACTMain(unittest.TestCase):
  def test_paper_act(self):
    ### An example of a traditional ACT process.

    embed = 5
    batch = 4

    shape = [batch, embed]

    # Define mock features
    make_state =  lambda : random.standard_normal(shape)
    make_output = lambda : random.standard_normal(shape)
    make_probs = lambda : 0.5* np.ones([batch])

    #Run act loop
    act_manager = ACTController(1, ["state", "output"])
    while not act_manager.is_halted:

      state = make_state()
      output = make_output()
      probs = make_probs()

      act_manager.buffer_tensors("state", state)
      act_manager.buffer_tensors("output", output)
      act_manager.update_channels(probs)

    # Check shapes
    self.assertTrue(act_manager["state"].shape == (batch, embed))
    self.assertTrue(act_manager["output"].shape == (batch, embed))
    self.assertTrue(act_manager.residuals.shape == (batch,))

  def test_nested_data(self):
      """ An example of an act structure handling nested data and more complex state"""

      # Nested datastructures can be incorporated into the
      # act process. When this happens,
      embed = 5
      batch = 4

      shape = [batch, embed]

      # Define mock features

      make_control_state =  lambda : random.standard_normal(shape)
      make_memory_state = lambda : {"LSTM" : ( random.standard_normal(shape),
                                               random.standard_normal(shape)
                                              ),
                                    "RecurrentAttn" : [
                                        random.standard_normal(shape),
                                        random.standard_normal(shape)
                                      ]
                                    }
      make_output = lambda : random.standard_normal(shape)
      make_probs = lambda : 0.5* np.ones([batch])

      # Run ACT loop
      act_controller = ACTController(1, ["control_state", "memory", "output"])
      while not act_controller.is_halted:

        control_state = make_control_state()
        memory = make_memory_state()
        output = make_output()
        probs = make_probs()

        act_controller.buffer_tensors("control_state", control_state)
        act_controller.buffer_tensors("memory", memory)
        act_controller.buffer_tensors("output", output)
        act_controller.update_channels(probs)

  def test_multidimensonal_batch(self):
    """ A batch can be multidimensional, such as in an ensemble. Test it"""

        ### An example of a traditional ACT process.

    embed = 5
    batch = 4
    ensemble= 6
    batch_dims = 2

    shape = [batch, ensemble, embed]

    # Define mock features
    make_state =  lambda : random.standard_normal(shape)
    make_output = lambda : random.standard_normal(shape)
    make_probs = lambda : 0.5* np.ones([batch, ensemble])

    #Run act loop
    act_manager = ACTController(batch_dims, ["state", "output"])
    while not act_manager.is_halted:

      state = make_state()
      output = make_output()
      probs = make_probs()

      act_manager.buffer_tensors("state", state)
      act_manager.buffer_tensors("output", output)
      act_manager.update_channels(probs)

    # Check shapes
    self.assertTrue(act_manager["state"].shape == (batch, ensemble, embed))
    self.assertTrue(act_manager["output"].shape == (batch, ensemble, embed))
    self.assertTrue(act_manager.residuals.shape == (batch, ensemble))

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=import-error

import unittest
import numpy as np
from numpy import random
from trax_act_controller import ACTController


class TestACTConstructor(unittest.TestCase):
    def test_default_setup(self):
        """Test that the default setup makes the correct channels"""

        # A simple test for if the channels are created correctly.
        instance = ACTController(1, ("test", "test2"))

        self.assertTrue(instance.num_batch_dims == 1)
        self.assertTrue(instance.batch_shape is None)
        self.assertTrue(instance.channel_names == ("test", "test2"))
        self.assertTrue(instance.get_channel("test", force = True) is None)

    def test_setup_accumulator_manual(self):
        """ Test that setting the accumulators to something manually works """

        # One can manually create the accumulators used in the channels. This verifies everything
        # sets properly, and parameters are updated.

        init = {"test" : random.standard_normal([10, 25]), "test2" : None}
        instance = ACTController(1, init)

        self.assertTrue(instance.batch_shape == (10,)) # This should trigger the batch shape logic
        self.assertTrue(np.all(instance.get_channel("test", force=True) == init["test"]))
        self.assertTrue(np.all(instance.get_channel("test2", force=True) == init["test2"]))

    def test_setup_probability_manual(self):
        """ Test that manually specifying the probability works"""

        init = ["test"]
        prob = np.array([[0.5],[0.7]])
        instance = ACTController(1, init, prob)

        self.assertTrue(instance.channel_names == ("test",))
        self.assertTrue(instance.batch_shape == (2,))

    def test_setup_accumulator_other(self):
        """Test that manual set, but with None everywhere, works"""

        # Some automatic construction logic might, for some reason, use this.
        init = {"test" : None}
        instance = ACTController(num_batch_dims = 1,
                                 channels = init)

        # No batch shape information captured yet, as that is grabbed as a lazy operation.
        self.assertTrue(instance.batch_shape is None)

    def test_setup_multiple(self):
        """Test that setup with multiple cases works"""

        # So long as batch dimensions match, this should be legal.
        init = {"test" : random.standard_normal([10, 25]), "test2" : random.standard_normal([10, 14, 7])}
        instance = ACTController(1, init)

        self.assertTrue(instance.batch_shape == (10,)) # This should trigger the batch shape logic
        self.assertTrue(np.all(instance.get_channel("test", force=True) == init["test"]))
        self.assertTrue(np.all(instance.get_channel("test2", force=True) == init["test2"]))



class TestACTConstructorErrors(unittest.TestCase):
    def test_setup_unequal_batch_dims(self):
        """Test we detect batch dims that are not the same"""

        # In this example, 5 and 10 are not the same, and thus invalid
        init = {"test" : random.standard_normal([5, 15]), "test2" : random.standard_normal([10, 30])}
        with self.assertRaises(ValueError):
            ACTController(1, init) # pylint: disable=pointless-statement

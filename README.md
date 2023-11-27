# trax-act

Adaptive Computation Time (ACT) Controller for Dynamic Computation Processes
The ACTController class in this package provides an implementation of the Adaptive Computation Time (ACT) technique, primarily used in neural networks to adjust computation steps dynamically based on input data complexity.

# Key Features

* Multiple Channels: Manages data across multiple channels, each accumulating data independently.
* Nested Data Structures: Handles dictionaries, lists, and tuples within channels.
* Halting Statistics: Monitors iterations required for halting.
* Batch Dimensions Support: Enables parallel processing of multiple data sets.
* Error Handling: Comprehensive error handling with detailed messages.
* Lazy Batch Shape Inference: Infers batch shape from the first tensor data.

# Installation

To install the package, use the following command:

```bash
pip install trax_act
```

# Usage

The documentation on the class is fairly thorough. In general, however

* Initialize the class with batch dimensions and channel names.
* Use the class's .is_halted method to run a while loop
* Buffer tensors into channels using buffer_tensors.
* Update channels with halting probabilities using update_channels.
* Wait until halted
* Access accumulated data using get_channel or channel indexing.

# Examples

Basic usage example. This is a reproduction of traditional Adaptive Computation Time
as would be found in the paper. 

```python
from trax-act import ACTController
import numpy as np

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

Advanced case, with mock data in "vector" and "matrix". You could, however, in theory accumulate anything, like
state across an entire model

```python
from trax-act import ACTController
import numpy as np

act_controller = ACTController(num_batch_dims=1, channels=["composite_data"])
nested_data = {"vector": np.array([1.0, 2.0]), "matrix": np.array([[1, 2], [3, 4]])}
while not act_controller.is_halted:
    act_controller.buffer_tensors("composite_data", nested_data)
    halting_probabilities = np.array([0.7])
    act_controller.update_channels(halting_probabilities)

accumulated_data = act_controller.get_channel("composite_data")

```
# Contribution Guidelines

To maintain the quality and consistency of the codebase, we ask contributors to follow these guidelines.

## Pull Requests
When submitting a pull request, please ensure the following:

* **Thorough Comments**: Your code should be well-commented to explain your logic and decisions.
* **Write Tests**: Include tests for your new code. This ensures reliability and helps others understand your contributions.
* **Pass All Tests**: All existing and new tests should pass before your pull request can be accepted.
* **Code Review**: Be open to feedback and make necessary revisions during the code review process.

## Reporting Issues or Bugs
For reporting bugs, please create a detailed issue on GitHub with the following information:

* Steps to reproduce the bug.
* The expected outcome and the actual outcome.
* Any relevant code snippets or error messages.
* Information about your operating system and environment, especially to ensure Unix compatibility.

## Feature Requests
We welcome new ideas and suggestions. To request a feature:

* Check if the feature has already been requested or exists.
* If not, create a new issue, clearly describing the feature and its potential benefits.
* Provide any thoughts on how the feature might be implemented.

# LicenseThis project is licensed under the MIT License.

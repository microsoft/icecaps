## Overview

Microsoft Icecaps is an open-source toolkit for building neural conversational systems. Icecaps provides an array of tools from recent conversation modeling and general NLP literature within a flexible paradigm that enables complex multi-task learning setups. 

Icecaps is currently on version 0.1.2. We have a number of features in the works for version 0.2, including new models and new data processing functionality. Please be aware that Icecaps 0.2 will not guarantee backwards compatibility and may cause breaking changes.


## Dependencies

Icecaps is intended for Python environments and is built on top of TensorFlow. We recommend using Icecaps in an Anaconda environment with Python 3.7. Once you have created an environment, run the following command to install all required dependencies:
``` python
pip install -r requirements.txt
``` 
If your machine has a GPU, we recommend you instead install from `requirements-gpu.txt`.


## Tutorials

We will be publishing tutorials soon on how to build various kinds of conversational scenarios in Icecaps.
Until then, we encourage you to look at the scripts in the `examples/` directory and use them as templates.

`examples/train_simple_example.py` is our "Hello World" script: 
it builds a simple seq2seq training scenario while demonstrating the basic five-phase pattern that Icecaps scripts follow.

`examples/train_persona_mmi_example.py` presents a more complex system that introduces component chaining and multi-task learning,
the core aspects of Icecaps' architecture.

Finally, `examples/data_processing_example.py` gives an example of how to convert a raw text dataset to TFRecord files, 
which Icecaps uses to feed its data pipelines during training.


## Pre-Trained Systems

We plan to add pre-trained systems based on cutting-edge conversational modeling literature to Icecaps in the future.
We had hoped to include these systems with Icecaps at launch. However, given that these systems may
produce toxic responses in some contexts, we have decided to explore improved content-filtering techniques before
releasing these models to the public. Follow this repository to stay up to date with new pre-trained system releases.


## Resources

Visit our homepage here: https://www.microsoft.com/en-us/research/project/microsoft-icecaps/

View our system demonstration paper from ACL 2019 here: https://www.aclweb.org/anthology/P19-3021


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


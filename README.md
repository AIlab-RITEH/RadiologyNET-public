# RadiologyNET Toolset - Public
API system and modules for different types of DICOM image manipulation and processing.


## Installation
1. Clone this repository.
2. `cd` into the directory where you cloned this repository. Make sure you're in the same folder where `setup.py` is located.
3. Run `pip install .`. This command will use the `setup.py` script to install the `radiologynet-toolset` package. 
    * If the command `pip install .` fails with error: 
        ```
            Complete output (6 lines):
        usage: setup.py [global_opts] cmd1 [cmd1_opts] [cmd2 [cmd2_opts] ...]
            or: setup.py --help [cmd1 cmd2 ...]
            or: setup.py --help-commands
            or: setup.py cmd --help
        
        error: invalid command 'bdist_wheel'
        ----------------------------------------
        ERROR: Failed building wheel for radiologynet-toolset
        ```
        then please make sure you run `pip install wheel` before running `pip install .` again.
4. On GTX 3090 GPU, CUDA verison used is 11.3. To install and use `torch`, install the proper version using the command:
    ```
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
    ```
    For more details on CUDA and torch versions and how to install torch, see [this link](https://pytorch.org/get-started/locally/).

## Organization
Currently, the package is divided into two main subpackages:
1. The `tools` subpackage is the core of the `radiologynet-toolset` package, it contains various utilities such as:
    * Raw data reading & parsing
    * Data vizualitaion, statistics and analysis
    * DICOM image extraction and conversion
    * ...
2. The `learn` subpackage is dedicated to machine learning *et cetera*


## Code Guidelines
Please, follow [PEP8](https://pep8.org/) as often and as much as possible.

When implementing new features, functions and utilities, please use docstrings to show specifications of your code. See [Google Docstring practices](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for more guides. Example of a docstring:
```
def some_function(arg1, arg2='default', arg3=None):
    """
    One-line description of what this function does.

    If this function needs a more elaborate description, provide it below the one-line description
    (in one or more paragraphs). Markdown syntax (such as bullet lists and headings) is supported.
    There should be at least one blank line between the long description and the one-liner.

    Args:
        arg1 (type) - Describe what this argument is,
            what's its desired type (`type`), what should it contain.
        
        arg2 (str, optional) - Describe what this argument is,
            what's its desired type (`str`), what should it contain.
            This parameter is optional. Defaults to 'default'.

        arg3 (SomeClass, optional) - Describe what this argument is,
            what's its desired type (`SomeClass`), what should it contain.
            This parameter is optional. Defaults to None.

    Returns:
        Describe what the return value is, what's its shape/type. This section can span multiple lines
        and have multiple paragraphs.

    Raises:
        ValueError: what is the cause for ValueError? Describe why it could be thrown.

        AttributeError: what is the cause for AttributeError? Describe why it could be thrown.
    """
    # ... some implementation

    return result
```

By executing the command `help(some_function)` in Python, the docstring will be printed out.

**Please don't forget to use docstrings. They are of great help to everyone involved in the development of this package.**

To provide documentation for different modules, please use docstrings in the `__init__.py` file of the respective module.

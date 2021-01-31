# Natural language processing: project

## Contents

This project will serve as a demonstration of the **PRADO** architecture by [Kaliamoorthi et al., 2019](https://www.aclweb.org/anthology/D19-1506). The PyPi packages used to implement the functionality are also made by me, albeit on a personal GitHub and PyPi account. The results are evaluated on the simple [IMDB 50k dataset](https://ai.stanford.edu/~amaas/data/sentiment/), by [Maas et al., 2011](http://www.aclweb.org/anthology/P11-1015).

## Installation

This uses **Python 3.8.5**. Other versions may work (but likely the minimum is **3.6** due to type hints), as may later versions (although for **3.9+** you'll have to use a different command to install **PyTorch**).

Assuming you use one of the [conda](https://docs.conda.io/en/latest/) flavours as a package manager, do:

- run `conda create -n env_name python=3.8.5` (skip this if you already have an environment ready)
  - your environment name can be whatever, but for this tutorial `env_name` will serve as a reference to whatever name you choose
- run `conda activate env_name`
- install **PyTorch 1.7.1** according to the instructions [here](https://pytorch.org/get-started/locally/) (skip this if **PyTorch 1.7.1** is already installed in your environment)
  - ex. for my Linux build the command was `conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch`
- position yourself in the root directory of the project (relative to the repository it's `./project`)
- run `pip install -r requirements.txt`

### Optional steps

If you want to run the Jupyter notebooks:

- run `conda install ipykernel` (skip this if you already have `ipykernel` installed in your environment)
- run `python3 -m ipykernel install --user --name=env_name`

Although you can run Jupyter without this due to it being in the `requirements.txt`, this is so you can use your environment to run the notebook.

## Documentation

The documentation, rather, the report can be found in the `docs` folder. If you found this repository useful, rather than citing the report cite this repository. After all, it's the implementation that counts here, the results should be attributed to the original architecture authors :D.

## Execution

You can use the scripts in the `scripts` folder by running `bash scripts/name-of-script.sh`. They run the python code and serve as templates for easier argument passing. Make sure that whatever you run, you run while in the root folder of the project (again, `./project`).

## Examples

All examples are within the `demo` folder, in the shape of Jupyter notebooks. Running a notebook is an idempotent operation, so there's no fear you'll break something if you run it several times.

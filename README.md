# `nnunet_fetalbrain`

Snakemake workflow for fetal bold brain segmentation

Training is best with a GPU, but inference can be done reasonably fast with CPU only.

### Step 1: Obtain a copy of this workflow

1. Create a new github repository using this workflow [as a template](https://help.github.com/en/articles/creating-a-repository-from-a-template).
2. [Clone](https://help.github.com/en/articles/cloning-a-repository) the newly created repository to your local system, into the place where you want to perform the data analysis.

### Step 2: Configure workflow

Configure the workflow according to your needs via editing `config.yml` file, specifically the paths to your nifti images.

### Step 3: Install python dependencies

You should install your dependencies in a virtual environment. Once you have activated your virtual environment, you can install the dependencies with `pip install .`

A recommended alternative that also takes care of creating a virtual environment is to use Poetry. On OSX or Linux can be installed with:
```
curl -sSL https://install.python-poetry.org | python3 -
```

Once you have poetry installed you can simply use the following to install dependencies into a virtual environment, then activate it:

```
cd nnunet-fetalbrain
poetry install
poetry shell
```

### Step 4: Execute workflow

To run inference on your test datasets, use:

    snakemake  all_test --cores all


By default, the trained model in the config will be downloaded and applied.

If you want to train a new model, set the `use_downloaded` config variable to one that is not in the `download_model`, then use:

    snakemake all_train --cores all 



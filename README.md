# Final project of Reproducible reasearch course on FEEC - 1st 2020

Repository to the final project of Computational Reproducible Research course from Unicamp 1st semester master course.


# Repository structure

The files are organized as follow:

  - /data - Dataset used
  - /deliver - The executable paper and experiment's codes
  - /dev - Codes and notebooks used during the experiment
  - /figures - Figures used on research
  - /utils - Some recommendations about the process of research in a Best Practices file

# Requisites

To run our "Reproducible Paper.ipynb" in "/deliver" folder you must first install the pre requisites:

## Ubuntu

You can run the file in ubuntu by installing the dependencies in "requeriments.txt"

So make sure you have python3 and pip3 installed

Git clone this repository

Go to the folder in terminal

Type

<code> python3 -m pip install -r requeriments.txt </code>

Then you should be able to open jupyter and run the notebook.

## Windows 

Install Anaconda with python 3 and make sure you have pip installed

Then clone the repository

Open the Anaconda Prompt

Go to the repository folder, where there is the "environment.yml"

Type "conda env create -f environment.yml" and then "conda activate reproducibility"

Done. You should be able to run the reproducible paper.

# How to run

To run the entire pipeline you must run all cells of the following notebooks inside the conda env:

<code> /dev/Training Language Model.ipynb </code>

<code> /deliver/Generate_dependencies.ipynb </code>

Finally:

<code> /deliver/Reproducible Paper.ipynb </code>


# Paper

By now the paper is under writing. The main sections and an initial text about dependent methods are already on the document.



# Licenses

  - Data: CC BY-NC-SA 3.0
  - Code: GNU GPL 3.0

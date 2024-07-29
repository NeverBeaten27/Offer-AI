# Offer AI

## Introduction

Offer AI is a Python-based application that extracts data from `.docx` files, fine-tunes a language model, and generates offers based on customer requirements. This guide will walk you through setting up the development environment in PyCharm, installing the necessary dependencies, and running the application.

## Prerequisites

- PyCharm IDE
- Python 3.12

## Setup Instructions

### 1. Install Python 3.12

If you don't have Python 3.12 installed, download and install it from the official Python website: [Python 3.12](https://www.python.org/downloads/).

### 2. Install PyCharm

Download and install PyCharm from the official JetBrains website: [PyCharm](https://www.jetbrains.com/pycharm/download/).

### 3. Configure Python Interpreter in PyCharm

1. Open PyCharm and create a new project or open an existing one.
2. Go to `File > Settings` (or `PyCharm > Preferences` on macOS).
3. Navigate to `Project: <your_project_name> > Python Interpreter`.
4. Click the gear icon and select `Add`.
5. Choose `System Interpreter` and select the Python 3.12 executable.
6. Click `OK` to save the settings.

### 4. Install Required Libraries

Open the terminal in PyCharm and run the following commands to install the necessary libraries:


pip install transformers datasets torch python-docx
pip install transformers[torch]

Example:

######################### Welcome to Offer AI #########################

################################ Menu #################################
1. Extract data
2. Train AI
3. Generate offer
4. Help
5. Exit

Please input menu number: 1
Please input the folder containing the .docx files: /path/to/docx_folder
Data extracted successfully.

Please input menu number: 2
Please input the number of training epochs: 3
Template created successfully.

Please input menu number: 3
Please input the customer's requirements: Introducerea clientilor in baza de date , emiterea de documente in mod automat/client, trimiterea lor cu posibilitatea de a semna digital in sistemul DocuSign.
Generated Offer:
...

Please input menu number: 5

So far the output can only be seen in the generation_log.txt.txt, but it generates some funky stuff, maybe because i didn't train it for enough generation, maybe the attention mask isn't correctly implemented and i did not get to implement the validation function.

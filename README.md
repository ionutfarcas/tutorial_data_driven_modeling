# Data-driven Model Reduction in the Time and Frequency Domains

Minitutorial [MT6](https://meetings.siam.org/sess/dsp_programsess.cfm?SESSIONCODE=82504)/[MT7](https://meetings.siam.org/sess/dsp_programsess.cfm?SESSIONCODE=83058), SIAM Conference on Computational Science and Engineering ([CSE25](https://www.siam.org/conferences-events/siam-conferences/cse25/))\
[Ionut-Gabriel Farcas](https://scholar.google.com/citations?user=Cts5ePIAAAAJ), Virginia Tech\
[Shane A. McQuarrie](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ), Sandia National Laboratories\
[Steffen Werner](https://scholar.google.com/citations?user=F2v1uKAAAAAJ), Virginia Tech\
March 2025, Fort Worth, TX

## Contents

This minitutorial is presented in two parts.

- [**TimeDomain/**](/TimeDomain/) contains data and examples of data-driven model reduction when observations of the system state are available.
- [**FrequencyDomain/**](./FrequencyDomain/) contains data and examples of data-driven model reduction when frequency input-output observations are available.

See [slides.pdf](./slides.pdf) for the presentation slides.

## Local Installation with Git

To get started, clone this repository with [`git`](https://git-scm.com/) to get a local copy of the code.

If you have GitHub `ssh` keys set up, clone with ssh.

```shell
git clone git@github.com:ionutfarcas/tutorial_data_driven_modeling.git
```

Otherwise, clone with https.

```shell
git clone https://github.com/ionutfarcas/tutorial_data_driven_modeling.git
```

The code examples are written in Python.
We recommend creating a new `conda` environment and installing the prerequisites listed in [requirements.txt](./requirements.txt).

```shell
# Deactivate any current environments.
$ conda deactivate

# Create a new environment.
$ conda create -n cse-minitutorial python=3.12

# Activate the new environment.
$ conda activate cse-minitutorial

# Install required packages.
(cse-minitutorial) $ pip install -r requirements.txt
```

## Google Colab

The code examples can be also accessed on Google Colab here:

- [TimeDomain/CompressibleEuler1D/demo.ipynb](https://colab.research.google.com/drive/1YtNnn6YFU3EhOdEd1i_Jqdx6bXovDAif?usp=sharing)
- [TimeDomain/VortexShedding2D/demo.ipynb](https://colab.research.google.com/drive/1JF2xXvCf8zJH4VoIAzWNHXS3BbBBdImk?usp=sharing)
- [FrequencyDomain/MassSpringDampler/demo.ipynb](https://colab.research.google.com/drive/1N4gu_tS1WWnzlhZJkrG6Lil4BCpu-siK?usp=sharing)
- [FrequencyDomain/ThermalDiffusion/demo.ipynb](https://colab.research.google.com/drive/14urxp4piIsUyZ4ll0R1wLAtVJ83XzVMR?usp=sharing)
- [FrequencyDomain/PorousBone/demo.ipynb](https://colab.research.google.com/drive/1GiHAR4CY0DRtaUg_u53AeDCERoUiWNxx?usp=sharing)

# Eye Tracking Cooking Recipes

## Set up environment
- install poetry
- run `poetry update`
- run `poetry install`
- open jupyter and select correct kernel, which can be found by running `poetry env info`

Using anaconda or miniconda is recommended.

## Create Plots
1. Open Jupyter notebook, either in web or via VSCode extension
2. Open `set_eye_tracking_cooking_recipes\presentation_notebooks\presentation.ipynb`
3. Press `Run All`

## Project structure
- data: contains the raw data from Tobii and the survey
- entities: data classes are defined to structure and extract relevant information from the raw data
- presentation_notebooks: Notebooks used for presentation
- test_notebooks: Just testing notebooks for own tests (can contain bugs)
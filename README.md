# Image Captioning

* To run the model with the default configurations (512 hidden dimensions, 300 embedding size, stochastic caption generation, 0.1 temperature, learning rate = 5e-4), run `python3 main.py`
Tune the hyperparameters by editing the `default.json` file:
- `num_epochs` to set the number of epochs
- `learning_rate` to change the learning rate
- `hidden_size` to set the number of hidden units in the LSTM
- `embedding_size` to set the embedding size for the words
- `model_type` set to "LSTM" to use LSTMs, else set to "RNN" to use Vanilla RNNs
- `max_length` to set the maximum size of a generated caption
- `deterministic` set to `True` to use deterministic caption generation and `False` for stochastic caption generation
- `temperature` to set the temperature for stochastic caption generation when using weighted softmax   

## Usage

* Define the configurations for your experiment. See `default.json` to see the structure and available options. You are free to modify and restructure the configuration as per your needs.
* Implement factories to return project specific models, datasets based on config. Add more flags as per requirement in the config.
* Implement `experiment.py` based on the project requirements.
* After defining the configuration (say `my_exp.json`) - simply run `python3 main.py my_exp` to start the experiment
* The logs, stats, plots and saved models would be stored in `./experiment_data/my_exp` dir. This can be configured in `contants.py`
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training pr evaluate performance.

## Files
- main.py: Main driver class
- experiment.py: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- dataset_factory: Factory to build datasets based on config
- model_factory.py: Factory to build models based on config
- constants.py: constants used across the project
- file_utils.py: utility functions for handling files 
- caption_utils.py: utility functions to generate bleu scores
- vocab.py: A simple Vocabulary wrapper
- coco_dataset: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset
- get_datasets.ipynb: A helper notebook to set up the dataset in your workspace

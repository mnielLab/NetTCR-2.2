# NetTCR-2.2 - Sequence-based prediction of peptide-TCR interactions using paired chain data
NetTCR-2.2 is a deep learning model used to predict TCR specificity. Like NetTCR-2.1 [1], NetTCR-2.2 uses convolutional neural networks (CNN) to predict whether a given TCR binds a specific peptide. The NetTCR-2.2 publication is currently unpublished, but the NetTCR-2.1 publication is available at https://www.frontiersin.org/articles/10.3389/fimmu.2022.1055151/full.

The scripts in this repository allow training and testing of models in three different modes, namely `pan`, `peptide` and `pretrained`. 

The pan-specific mode (`src/train_nettcr_2_2_pan.py`) allows for training and predictions on multiple different peptides at same time, and can in theory also be used to predict binding for peptides not in the training data, though accuracy may be quite poor for peptides distant to the training peptides. 

The peptide-specific mode (`src/train_nettcr_2_2_peptide.py`) is trained on a single peptide, one at a time, and is thus very similar to the approach in NetTCR 2.1. 

The pretrained mode (`src/train_nettcr_2_2_pretrained.py`) uses a combination of the pan- and peptide-specific approaches, by first pre-training on a dataset consisting of multiple different peptides. Following this, the model is trained again on a single peptide, one after the other, producing a peptide-specific model for each peptide in the training data.

The model architectures were built in Keras [2], and are located in `src/nettcr_archs.py`.

## Data
The folder `data/`contains the data used to train/validate/test NetTCR-2.2. Here, the `nettcr_2_2_full_dataset.csv` file consists of the full dataset, which the evaluation was performed on, while `nettcr_2_2_limited_dataset.csv` is a limited version of this dataset with potential outliers removed, which was the dataset that the final models were trained on. The positive data was retrieved from IEDB, VDJdb, and 10X genomics datasets. The negative data was generated by mismatching positive TCRs within each partition with other peptides that had a Levenshtein distance of at least 3 from the peptide in question (denoted as `peptide_swapped`). 

The redundancy in the dataset was reduced using Hobohm1 algorithm [3], using the kernel similarity [4] measure and a similarity threshold of 0.95 for the CDR3 sequences. Thus training, validation and test dataset do not share similar TCR sequences (up to 0.95 similarity threshold).

In addition to the training data, a set of negative controls from the IMMREP 2022 benchmark [5] is also included under `data/negative_controls.csv`. These were used for converting prediction scores into percentile rank scores, by comparing the predictions for binding between peptides and the negative control TCRs, to the TCRs in the test dataset.

The `data/leave_most_out` folder contains datasets were the peptides with at least 100 positive observations in `nettcr_2_2_limited_dataset.csv` were downsampled to 5, 10, 15, 20, 25, 50 and 100 positives, respecively, while keeping a positive to negative ratio of 1:5.

The `data/IMMREP` folder contains datasets from the IMMREP 2022 benchmark [5], which was used for comparing NetTCR-2.2 to other models. The observations for each peptide was merged into a single train file (`data/IMMREP/train/all_peptides.csv`) and test file (`data/IMMREP/test/all_peptides.csv`), and the observation in the train file were randomly partitioned into 5 partitions.

`data/IMMREP/test/rank_test.csv` contains the combinations between all peptides and all positive TCRs in the test data, and was used to assess the average rank given by each model.

Finally, a new training dataset (`data/IMMREP/train/all_peptides_redundancy_reduced.csv`)was constructed based on the IMMREP 2022 benchmark training, which was subjected to the same redundancy reduction as our primary dataset (`nettcr_2_2_full_dataset.csv`), since we observed issues with redundant observations. Additionally, swapped negatives were generated within each partition, rather than across partitions. A new set of redundancy reduced negative controls (95% threshold) was also used for this dataset. The final ratio between positives, swapped negatives and negative controls were the same as the original dataset (1:3:2).

## Environment setup with conda

First, ensure that miniconda/anaconda is installed (see `https://docs.conda.io/projects/miniconda/en/latest/` for more info).

Following this, the conda environment for NetTCR-2.2 can be installed by running `conda env create -f environment.yml` from the `NetTCR-2.2` folder. This will create a conda environment called `nettcr_2_2_env` with the necessary dependencies.

To load this environment afterwards, run the command `conda activate nettcr_2_2_env`.

## Network training

The input datasets should contain the peptide sequence, as well as the sequences of all six CDRs. The columns in the input data should be `peptide`, `A1`,`A2`,`A3`, `B1`, `B2`, `B3`, and the input files should be comma-separated (See `data/examples/train_example.csv` as an example). Additionally, the files used for training and validation should have a binary column called `binder`, where `1` indicates binding to a TCR, while `0` indicates no binding.

The inputs files for the training scripts are the training dataset and the validation data (which is used for early stopping).

Example:

`python src/train_nettcr_2_2_pan.py --train_data data/examples/train_example.csv --val_data data/examples/validation_example.csv --outdir <outdir> --model_name <model_name>`

This will generate and save a `<model_name>.h5` and `<model_name>.tflite` file for the trained model under `<outdir>/checkpoint`. The `.h5` can be used if the model should be re-trained, whereas the `.tflite` file is used for fast predictions.

The other input arguments to the script are `--dropout_rate`, `--learning_rate`, `--patience`, `--epochs`, `batch_size`, `--verbose`, `--seed` and `--inter_threads`.

## Network testing 
The `src/predict.py` script can be used to make predictions on TCRs, using one of the trained models.

Example:

`python src/predict.py --test_data data/examples/test_example.csv --outdir <outdir> --model_name <model_name> --model_type {pan/peptide/pretrained}`   

This will generate and save a `<model_name>_prediction.csv` file with the predictions, which is saved in the specified output directory. 

## Trained Models
The folder `models` contains the final models from the NetTCR 2.2 publication (currently unpublished), along with the predictions used in the publication (`cv_pred_df.csv`). Additionaly, a `negative_controls` is included, which contains the predictions on the negative controls from the IMMREP 2022 benchmark [5], which is used for assigning a percentile rank to each observation.

The models in this folder can be used for predictions in the same way as the models trained by the user. Example:

`python src/predict.py --test_data data/examples/test_example.csv --outdir models/nettcr_2_2_peptide --model_name "t.1.v.2" --model_type peptide`   

### References
[1] Montemurro, Alessandro, et al. "NetTCR-2.1: Lessons and guidance on how to develop models for TCR specificity predictions." *Frontiers in Immunology* Volume 13 (2022).

[2] Chollet, François, et al. (2015) "Keras" https://github.com/fchollet/keras

[3] Hobohm, Uwe, et al. "Selection of representative protein data sets." *Protein Science* 1.3 (1992): 409-417.

[4] Shen, Wen-Jun, et al. "Towards a mathematical foundation of immunology and amino acid chains." *arXiv preprint arXiv:1205.6031* (2012).

[5] Meysman, Peter, et al. "Benchmarking solutions to the T-cell receptor epitope prediction problem: IMMREP22 workshop report." *ImmunoInformatics* Volume 9 (2023).


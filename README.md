# Pairwise Preference Multileave
This repository contains the code used for the experiments in "Sensitive and Scalable Online Evaluation with Theoretical Guarantees" published at CIKM 2017.

Reproducing Experiments
--------
Recreating the results in the paper for a single dataset (for instance NP2003) can be done with the following command:
```
python scripts/CIKM2017.py --n_proc 1 --click_models inf nav per --n_impr 10000 --data_folders NP2003 --n_rankers 15 --print_output --log_folder testoutput/logs/ --average_folder testoutput/average --output_folder testoutput/fullruns/ --n_runs 125 --print_freq 500
```
This runs all methods except for Optimized Multileaving since this requires [Gurobi](http://www.gurobi.com/) to be installed. 
It is up to the user to download the datasets and link to them in the [dataset collections](utils/datasetcollections.py) file.
The output folders including the folders where the data will be stored (in this case testoutput/fullruns/2003_np/) have to exist before running the code, if folders are missing an error message will indicate this.
Speeding up the simulations can be done by allocating more processes using the n_proc flag.

Citation
--------

If you use this code to produce results for your scientific publication, please refer to our CIKM 2017 paper:

```
@inproceedings{Oosterhuis2017OnlineEvaluation,
  title={Sensitive and Scalable Online Evaluation with Theoretical Guarantees},
  author={Oosterhuis, Harrie and de Rijke, Maarten},
  booktitle={CIKM},
  volume={2017},
  year={2017,
  organization={ACM}
}
```

License
-------

The contents of this repository are licensed under the [MIT license](LICENSE). If you modify its contents in any way, please link back to this repository.

# SRRNet
This is the PyTorch implementation of SSVEP Response Regression Network (SRRNet).

# Prepare the environment
Please create an anaconda virtual environment by:

`$ conda create --name srrnet python=3.8`

Activate the virtual env by:

`$ conda activate srrnet`

Install the requirements by:

```pip3 install -r requirements.txt```


# Run the code
## Prepare the datasets
Please download the Benchmark dataset and the BETA dataset at [here](https://bci.med.tsinghua.edu.cn/) and place them as follows:
```
father_folder/
├── Code/
│   ├── SRRNet/
│   └── ...
└── data/
    ├── ssvep_benchmark_2016/
        ├── data_sum/
            ├── Freq_Phase.mat
            ├── S1.mat
            ├── S2.mat
            └── ...
    └── ssvep_beta/
        ├── data/
            ├── S1.mat
            ├── S2.mat
            └── ...
```



## Run
To run the code, please type the following command in the terminal:

`$ python main_fold1.py --unseen-num 8 --dataset benchmark --subjects 35 --subject 1 --window 1.0 --model srrnet --spatialfilter trca --fb-num 5`

You may change or add command arguments, please refer to the function `set_args()` in `main_fold1.py` for details. The results are displayed via teriminal and saved in `Outputs/Results/CSV/`.

`ncpu_folds.py` and `ngpu_folds.py` use multiple CPUs/GPUs to run code with different conditions, e.g., diff subjs, unseens, datasets, etc. Note that they are only testified with the server in our lab.

# CBCR License
| Permissions | Limitations | Conditions |
| :---         |     :---:      |          :---: |
| :white_check_mark: Modification   | :x: Commercial use   | :warning: License and copyright notice   |
| :white_check_mark: Distribution     |       |      |
| :white_check_mark: Private use     |        |      |

# Others

Note: this is my first time sharing source code on GitHub. Feel free to contact me at maimam@sjtu.edu.cn if you have any question or suggestions :)

# Cite

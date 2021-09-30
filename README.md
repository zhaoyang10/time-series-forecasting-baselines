# Time series forecasting baselines
This is a repository for time series forecasting baselines.
We mainly focus on deep learning based methods, such as TCN, LSTM, Transformer based methods.

We aim to provide an environment for developers 
who need to reimplement SOTA paper works and to apply SOTA works into their own tasks.  


# Requirements
- [yacs](https://github.com/rbgirshick/yacs) (Yet Another Configuration System)
- [PyTorch](https://pytorch.org/) (An open source deep learning platform) 

# Table Of Contents
-  [In a Nutshell](#in-a-nutshell)
-  [In Details](#in-details)
-  [Future Work](#future-work)
-  [Contributing](#contributing)
-  [Acknowledgments](#acknowledgments)

# In a Nutshell   
In a nutshell here's how to use this code, so **for example** assume you want to run Informer to train ETT, so you should do the following:

```bash
bash bash_scripts/run_debug.sh
```

The algoroithm related codes are placed in **tsf_baselines.**
The config and scripts related codes are placed out of **tsf_baselines** directory.

# In Details
```
├── tsf_baselines - algorithm related codes.
│    └── config
│           └── defaults.py  - here's the default config file.
│
│    └── data  
│           └── datasets  - here's the datasets folder that is responsible for all data handling.
│           └── transforms  - here's the data preprocess folder that is responsible for all data augmentation.
│           └── build.py  		   - here's the file to make dataloader.
│           └── collate_batch.py   - here's the file that is responsible for merges a list of samples to form a mini-batch.
│
│     └── engine
│            └── trainer.py     - this file contains the train loops.
│            └── inference.py   - this file contains the inference process.
│
│     └── layers             - this folder contains any customed layers of your project.
│
│     └── solver             - this folder contains optimizer of your project.
│            └── build.py
│            └── lr_scheduler.py
│   
│     └── algorithm            - this folder contains any algorithm of your project.
│            └── build.py
│
│     └── modeling            - this folder contains any backbone model of your project.
│            └── informer
│     └── utils
│            └── logger.py
│            └── any_other_utils_you_need
│
├── configs  
│    └── informer
│           └── train_informer_v1.yml  - here's the specific config file for specific model or dataset.
│ 
├── tools                - here's the train/test model of your project.
│    └── train_net.py   - here's an example of train model that is responsible for the whole pipeline.
│ 
└── tests					- this foler contains unit test of your project.
     ├── test_data_sampler.py
```

# Results

All the results are the average of 5 runs in the same setting.

- Informer - univariate forecasting

| Dataset     | Setting |  MSE   |   MAE  |  RMSE |  MAPE  |  MSPE |
| :---        |  :----: | :----: |  :---: | :---: |  :---: | :---: |
| ETTm1       |    24   | 0.024  |  0.116 | 0.153 |  0.090 | 0.015 |
| ETTm1       |    48   | 0.056  |  0.182 | 0.237 |  0.139 | 0.033 |
| ETTm1       |    96   | 0.215  |  0.387 | 0.462 |  0.276 | 0.100 |
| ETTm1       |    288  | 0.362  |  0.523 | 0.601 |  0.380 | 0.182 |
| ETTm1       |    672  | 0.527  |  0.646 | 0.725 |  0.468 | 0.271 |

# Future Work
- Complete the Informer part and give a results table of this reimplemetation.
- Add other deep learning based algorithms.


# Contributing
Any kind of enhancement or contribution is welcomed.


# Acknowledgments
Great thanks to [Sidun Liu](https://github.com/Liu-SD).

Good competition and collaboration experience with him.
This repository is mainly built by us.

This project design is highly borrowed from [Deep-Learning-Project-Template](https://github.com/L1aoXingyu/Deep-Learning-Project-Template), [Domainbed](https://github.com/facebookresearch/DomainBed).

The algorithm code is highly borrowed from [Informer2020](https://github.com/zhouhaoyi/Informer2020)

 



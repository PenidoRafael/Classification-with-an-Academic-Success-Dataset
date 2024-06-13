## Objective ##
Our main objective is to predict the educational outcome of students based on various demographic, socio-economic, and academic features. Understanding these outcomes is essential for improving educational strategies and policies. We'll explore and compare different classification algorithms to effectively predict whether a student will graduate, drop out, or remain enrolled, and gain insights into the factors influencing these educational pathways.

## Strategy ##
The strategy guiding this competition will be to develop several good distinct classifiers and then create a strong classifier by ensembling them. Various techniques will be applied to achieve the best possible strong classifier.

## Structure ##
The repository structure is adapted to facilitate the development of ensemble strategies.

```bash
.
├── base_line.ipynb
├── ensemble.ipynb
├── models
│   ├── cat_1
│   │   ├── catboost_info
│   │   │   ├── catboost_training.json
│   │   │   ├── learn
│   │   │   │   └── events.out.tfevents
│   │   │   ├── learn_error.tsv
│   │   │   ├── time_left.tsv
│   │   │   └── tmp
│   │   ├── model.ipynb
│   │   └── oof
│   │       ├── fold_1.csv
│   │       ├── fold_2.csv
│   │       ├── fold_3.csv
│   │       ├── fold_4.csv
│   │       └── fold_5.csv
│   ├── lgbm_1
│   │   ├── model.ipynb
│   │   └── oof
│   │       ├── fold_1.csv
│   │       ├── fold_2.csv
│   │       ├── fold_3.csv
│   │       ├── fold_4.csv
│   │       └── fold_5.csv
│   └── xgb_1
│       ├── model.ipynb
│       └── oof
│           ├── fold_1.csv
│           ├── fold_2.csv
│           ├── fold_3.csv
│           ├── fold_4.csv
│           └── fold_5.csv
└── src
    ├── test
    │   ├── sample_submission.csv
    │   └── test.csv
    └── train
        ├── original.csv
        └── train.csv
```

Each individual model should have its respective folder within the models/ directory. Inside the individual model's folder, there should be a script that generates the out-of-fold predictions, which will be used in the ensemble training. Note also that there are no EDA notebooks here, as many people have already dedicated themselves to this task in the Kaggle competition itself. It is a good source of information and understanding of the data. Of course, if needed, there will be advances in this regard here.

## Next Steps ##
The competition ends on 06/30. Until then, we aim to climb the competition rankings significantly. It is necessary to first achieve good individual models; the current ones serve more as examples and do not include any hyperparameter optimization. Subsequently, there will also be an effort to select the features of the final ensemble (which individual models should be included).


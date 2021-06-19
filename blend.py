import numpy as np
import pandas as pd
ID_name = "id"
target_name = "stroke"

sub = pd.DataFrame(pd.read_csv(f'data/input/sample_submission.csv')[ID_name])
sub[target_name] = 0

base_subs = {
    "data/output/sub_lightgbm001.csv":1.,
    "data/output/sub_logistic_reg_000.csv":1.,
    "data/output/sub_nn3_000.csv":1.,
}

sum_weight = 0
for path,w in base_subs.items():
    tmp_sub = pd.read_csv(path)
    sub[target_name] += tmp_sub[target_name] * w
    sum_weight += w

sub[target_name] /= sum_weight


sub.to_csv(
    './data/output/sub_blend.csv',
    index=False
)

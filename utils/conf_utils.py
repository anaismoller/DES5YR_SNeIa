"""
Configuration utils for experiments/survey
"""

# seeds
list_seeds_set = {}
list_seeds_set[0] = [0, 55, 100, 1000, 30469]
list_seeds_set[1] = [30496, 49, 7510, 1444, 9]
list_seeds_set[2] = [1, 2, 303, 4444, 537]

all_seeds = list_seeds_set[0] + list_seeds_set[1] + list_seeds_set[2]

# sets
list_sets = [0, 1, 2]

# ensemble methods
# dic_sel_methods = {"predicted_target_S_":"single model","predicted_target_samepred_set":"target same pred", "predicted_target_average_set":"target average", "predicted_target_probability_average_set":"probability average"}
dic_sel_methods = {
    "predicted_target_S_": "single model",
    "predicted_target_average_target_set_": "ensemble (target av.)",
    "predicted_target_average_probability_set_": "ensemble (prob. av.)",
}


# Spectroscopic tags
spec_tags = {
    "Ia": [1, 101],
    "CC": [20, 21, 22, 23, 29, 32, 33, 39, 120, 121, 122, 129, 132, 133, 139],
    "SLSN": [41, 42, 43, 141, 142, 143, 66],
    "AGN": [80, 180],
    "TDE": [81],
    "Mstar": [82],
    "pec_Ia": [3, 4, 5],
}

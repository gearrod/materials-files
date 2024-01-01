# Imports
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def process_data():
    degrees = [45, 50, 60, 70, 75, 80, 90]

    partial_data = {
        "porus": {deg: None for deg in degrees},
        "non-porus": {deg: None for deg in degrees},
    }

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    extra_data_dir = os.path.join(data_dir, os.path.join("data_xls", "extra"))
    data_dir = os.path.join(data_dir, "data_csv")

    # Load porus partial_data
    for k, v in partial_data["porus"].items():
        partial_data["porus"][k] = pd.read_csv(
            os.path.join(
                data_dir, f"DATOS_POROS_PROPIEDADES_EFECTIVAS_AXIALES angulo{k}.csv"
            )
        )

    # Load non-porus partial_data
    for k, v in partial_data["non-porus"].items():
        partial_data["non-porus"][k] = pd.read_csv(
            os.path.join(data_dir, f"DATOS_PROPIEDADES_EFECTIVAS_AXIALES angulo{k}.csv")
        )

    # Load extra data from the data_xls/extra files
    xls_extra_files = os.listdir(extra_data_dir)  # Scan xls files in the folder

    for file in xls_extra_files:
        angle = int(file[42:44])

        # Add the extra data to non-porus datasets according to the angle
        partial_data["non-porus"][angle] = pd.concat(
            [
                partial_data["non-porus"][angle],
                pd.read_excel(os.path.join(extra_data_dir, f"{file}")),
            ],
            ignore_index=True,
        )

    # Concat the dataset grouped by
    middle_data = {
        "porus": pd.concat(partial_data["porus"].values(), ignore_index=True),
        "non-porus": pd.concat(partial_data["non-porus"].values(), ignore_index=True),
    }

    data = pd.concat(middle_data.values(), ignore_index=True)

    # Shuffles the dataset
    data = data.sample(frac=1, random_state=36).reset_index(drop=True)

    # Names the columns
    data.columns = [
        "angle",
        "p_matrix",
        "p_fiber",
        "radius",
        "ratio",
        "p11",
        "p22",
        "p12",
    ]

    data.drop(["radius"], axis=1, inplace=True)

    return partial_data, middle_data, data


# Define data in three global constants
PARTIAL_DATA, MIDDLE_DATA, GLOBAL_DATA = process_data()

assert GLOBAL_DATA.shape == (136680, 7)

# Split the validation data at 20% and store train and validation data in global constants
TRAIN_DATA, VALIDATION_DATA = train_test_split(
    GLOBAL_DATA, test_size=0.2, random_state=42
)

assert TRAIN_DATA.shape == (109344, 7)

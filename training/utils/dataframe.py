import pandas as pd


def create_model_dataframe():
    return pd.DataFrame({
        "k": [],
        "model": [],
        "c_npmi_train": [],
        "c_npmi_test": [],
        "diversity": [],
        "path": []
    })


def insert_line_in_model_dataframe(df, k, model_name, npmi_train, npmi_test, diversity, model_path):
    return df.append({
        "k": k,
        "model": model_name,
        "c_npmi_train": npmi_train,
        "c_npmi_test": npmi_test,
        "diversity": diversity,
        "path": model_path,
    }, ignore_index=True)

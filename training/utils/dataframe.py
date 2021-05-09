import pandas as pd


def create_model_dataframe():
    return pd.DataFrame({
        "k": [],
        "model": [],
        "c_npmi_train": [],
        "c_npmi_validation": [],
        "diversity": [],
        "path": [],
        "train_time_in_seconds": []
    })


def insert_line_in_model_dataframe(
    df, 
    k, 
    model_name, 
    npmi_train, 
    npmi_validation, 
    diversity, 
    model_path, 
    train_time_in_seconds,
):
    return df.append({
        "k": k,
        "model": model_name,
        "c_npmi_train": npmi_train,
        "c_npmi_validation": npmi_validation,
        "diversity": diversity,
        "path": model_path,
        "train_time_in_seconds": train_time_in_seconds,
    }, ignore_index=True)

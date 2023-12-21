"""Methods for joining up parts of a Reproducible Data Science workflow."""


def preprocess_data(df, filename):
    """Call all functions needed for preprocessing."""
    df = drop_duplicated_entries(df)
    df = drop_missing_values(df)
    # output
    if filename is not None:
        make_outputfile(df, filename)
    return df


def drop_duplicated_entries(df):
    """Return data with duplicate rows removed."""
    return df.drop_duplicates(inplace=False)


def drop_missing_values(df):
    """Return data with rows with missing values removed."""
    return df.dropna(axis=0, how='any', inplace=False)


def make_outputfile(df, filename):
    """Save result as interim data."""
    output_path = '../data/interim/' + filename
    df.to_csv(output_path, index=False)
    return

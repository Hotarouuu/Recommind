import numpy as np # linear algebra
import polars as pl # data processing, CSV file I/O (e.g. pd.read_csv)


def data_treatment(data, ratings):
    df_data = pl.read_csv(data)
    df_ratings = pl.read_csv(ratings)


    df_data = df_data[['Title', 'authors', 'categories', 'ratingsCount']]

    df_data = df_data.with_columns(
        pl.col("ratingsCount").fill_null(0),
        pl.col("categories").fill_null("No Category")
    )

    df_ratings = df_ratings[['Id','Title', 'User_id', 'review/score']]

    df_ratings = df_ratings.drop_nulls()

    df_merged = df_ratings.join(
        df_data.select(["Title", "authors", "categories", "ratingsCount"]),
        on='Title',
        how='left'
    )

    df_merged = df_merged.with_columns(  

        pl.col("categories").str.replace_all('[','',literal=True).str.replace_all(']','',literal=True),
        pl.col("authors").str.replace_all('[','',literal=True).str.replace_all(']','',literal=True),

    )

    return df_merged


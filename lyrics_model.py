import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import seaborn as sns
    import marimo as mo
    import pandas as pd
    import os
    import json
    return json, mo, np, os, pd, sns


@app.cell
def _():
    # lyrics_data |----- taylor_swift.json

    file_path = 'lyrics_data/taylor_swift.json'

    return (file_path,)


@app.cell
def _(file_path, pd):
    ts_df = pd.read_json(file_path, orient="index")
    return (ts_df,)


@app.cell
def _(pd):
    file_path = 'lyrics_data/taylor_swift.json'
    ts_df = pd.read_json(file_path, orient="index")
    return file_path, ts_df


@app.cell
def _(os, pd):

    def load_artist_lyrics(artist_name):
      """
      Loads lyrics data from a JSON file for a given artist.

      Args:
        artist_name: The name of the artist (e.g., 'billie_elish', 'rihanna', 'taylor_swift').

      Returns:
        A pandas DataFrame containing the lyrics data, or None if the file is not found.
      """
      file_path = os.path.join('lyrics_data', f'{artist_name}.json')
      try:
        df = pd.read_json(file_path, orient="index")
        return df
      except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return None

    # Example usage:
    billie_df = load_artist_lyrics('billie_elish')
    rihanna_df = load_artist_lyrics('rihanna')
    taylor_df = load_artist_lyrics('taylor_swift')

    return billie_df, load_artist_lyrics, rihanna_df, taylor_df


@app.cell
def _():
    return


@app.cell
def _(file_path, pd):
    ts_df = pd.read_json(file_path, orient="index")
    return (ts_df,)


if __name__ == "__main__":
    app.run()

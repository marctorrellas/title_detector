import pandas as pd
from IPython.core.display import display_html
from IPython.display import HTML, display


def notebook_setup():
    pd.set_option("max_rows", 1000)
    pd.set_option("max_columns", 1000)
    tmp = (
        "<style>"
        "     .output {"
        "          display: block;"
        "          align-items: center;"
        "          text-align: center;"
        "     }"
        "     .jp-Stdin-input {"
        "          width: 90%"
        "     }"
        "</style>"
    )
    display(HTML(tmp))


def display_side_by_side(*args):
    html_str = ""
    for df in args:
        html_str += df.to_html()
    html_str = html_str.replace("table", 'table style="display:inline"')
    display_html(html_str, raw=True)


def display_side_by_side_dflist(dfs, row_length=3):
    if len(dfs) <= row_length:
        display_side_by_side(*dfs)
    else:
        i = 0
        while i + row_length < len(dfs):
            display_side_by_side(*dfs[i : i + row_length])
            i += row_length
        if i < len(dfs):
            display_side_by_side(*dfs[i:])

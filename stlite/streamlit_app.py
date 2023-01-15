import random
import json
import time
import math

# import databutton as db
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import sys
from urllib.parse import quote
import urllib

if sys.platform == "emscripten":
    # running in Pyodide or other Emscripten based build
    PYODIDE = True
    # Print pyodide version
    print("Running in Pyodide")


def pyodide_fetch(
    url, method="GET", body: dict = {}, credentials="same-origin", headers={"Content-Type": "application/json"}
):
    # Function to open a URL in Pyodide
    from pyodide.http import open_url

    # url_quoted = quote(url, safe=",/")
    print(url)
    # https://pyodide.org/en/stable/usage/api/python-api/http.html#pyodide.http.open_url
    resp = open_url(f"https://thomasht86--mattemix-solver-wrapper.modal.run/{url}")
    # resp is a io.StringIO object. Convert to json
    text = resp.getvalue()
    return json.loads(text)


st.set_page_config(layout="centered", page_title="Mattemix puzzle solver 🎲⌛", page_icon="🎲")

# Get query parameters and display them
query_params = st.experimental_get_query_params()
if query_params:
    st.write(query_params)

with st.expander("About the game"):
    st.title("Mattemix")
    st.image(
        "https://lh3.googleusercontent.com/pDe2u87XMMDlbPnnrkKbFSlIxmMSptVR-5ZVtypH6g3MFbpYyjibqd4lrO3E4dUn8cazUUZvSwK0a0H5JlKT3IQr3grXAXE49fZ0EF-GjqL0uwZ9kSfRF288AIgHoW174-zPczuyyZEdmQDo4AOUgMzcXPG3SmxNAHnr0mBWNkU0V_SJhA2LG64dGeBlosgBl0CJGPksRu90VczUeNp1M8KRDJ0WCi42e9aIygh70MJv9aHFuAe7Ar2WfbQJ1iNJb4FJi3kmAJKW2UdCm-10VjluHBS-0OZ3jH-dT_a3NBEyhH21VvLbxqd63lVLSt_VtyajeGAqUKxgWYpwPyGzJlYhh2dCBWf0XS5CQIE-YF-5xlNi0eY8XLNe5tVl7_zgBJ9FmGxbQEWeFMzEdkE0q6oFJLfetUBAABiqa59KsyJ9A9K3ZKbD0stWeBdVnlcakU8iy7aV6RSGSMAfMEmZ3-xrqU0wvTB34EBLUrO2LsNBwfv1LPD76vhlg0GT3rpYdTGtCOLvd2SW0MXPRSZKeN9GK5s_n6w_m1XeS1gzsqX4hexX0Uctmu6hqWbmK4hhohMoFLcLw3muS0rLhnlYCly0eq80CVPKpqhy8t7tX9ZYN3Ph6wBBB6GX1yR-3q20OSlFcVdGYVltDl4Oc03ksh3fAfLd8c-unjlTaIADoPBz7nLYQfzO03S2FqP4N-aUUIlQUhloB49SpobHnTbtqPLMDCi91RCJdLeAKH45bfmXeQhcQnZU_dvvy3hk5GfSeE9xvj_7upFOKCApkZNdgaxzbcpVZfpI2vkx7le_orgufxkPnlujNpHA3x-XNfTMuh4YCE4v3r5A1gW_sxjE6n6MilMHj39JxaNE5PiDJTP47JoGnQxypXGpF9E1mT6mo8L84wlgBU_IgMWGRYFNKi5TFJsGhWHuwOw4TDbat3XXUA=w569-h439-no?authuser=0"
    )
    st.markdown(
        """
    Mattemix is a puzzle game where the goal is to place 14 dice on a board in 
    order to solve as many as possible of the 4 equations on the board.
    The score is calculated by the sum of the numbers of the dice that are succesfully
    placed on the board.
    """
    )


def get_roll():
    roll = np.random.choice([str(i) for i in range(1, 16)], 14, replace=True)
    return roll


def make_grid(cols, rows):
    grid = [0] * cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid


def write_grid(cols, rows, dice_array):
    grid = make_grid(rows, cols)
    for r in range(rows):
        for c in range(cols):
            num = (r * cols) + c
            grid[r][c].button(f"{dice_array[num]}", key=f"{r}_{c}+", disabled=True)
    return


def check_solution(dice_array):
    # Convert dice_array to a list of ints
    dice_array = [int(i) for i in dice_array]
    # Check if the solution is correct
    correct_equations = [False, False, False, False]
    score = 0
    eq1 = dice_array[0] + dice_array[1] == dice_array[2]
    if eq1:
        correct_equations[0] = True
        score += sum(dice_array[:3])
    eq2 = dice_array[3] - dice_array[4] == dice_array[5]
    if eq2:
        correct_equations[1] = True
        score += sum(dice_array[3:6])
    eq3 = (dice_array[6] / dice_array[7]) == ((dice_array[8] + dice_array[9]) | (dice_array[8]) - (dice_array[9]))
    if eq3:
        correct_equations[2] = True
        score += sum(dice_array[6:10])
    eq4 = (dice_array[10] * dice_array[11]) == (dice_array[12] + dice_array[13]) | (dice_array[12] - dice_array[13])
    if eq4:
        correct_equations[3] = True
        score += sum(dice_array[10:])
    st.session_state["best_score"] = score
    st.session_state["correct_equations"] = correct_equations
    return


def update_session_state(result_dict: dict):
    print(result_dict)
    for key, value in result_dict.items():
        if key in ["dice_array", "best_solution"]:
            # Don't copy these directly
            continue
        st.session_state[key] = value
    if len(result_dict["best_solution"]) == 14:
        st.session_state["dice_array"] = [str(i) for i in result_dict["best_solution"]]
    # TODO: Update the style based on correct equations
    return


def poll_results():
    res = pyodide_fetch(f"result/{st.session_state.call_id}")
    update_session_state(res)
    return


def get_solve_job(**kwarg_dict):
    st.session_state["time_left"] = kwarg_dict["timeout"]
    st.session_state["dice_array"] = kwarg_dict["dice_array"]
    kwarg_dict["dice_array"] = ",".join(kwarg_dict["dice_array"])
    qps = urllib.parse.urlencode(kwarg_dict)
    res = pyodide_fetch(f"solve?{qps}")
    st.session_state["call_id"] = res["call_id"]
    poll_results()
    return


# Initialization
if "dice_array" not in st.session_state:
    # Dice array is a list of 14 strings
    st.session_state["dice_array"] = [str(i) for i in range(1, 15)]
    check_solution(st.session_state["dice_array"])

with st.sidebar:
    with st.form("my_form"):
        st.markdown("## Set parameters for the Genetic Algorithm Solver 🧬")
        cols = st.columns(2)
        with cols[0]:
            pop_size = st.slider("pop_size", value=50000, max_value=100000, min_value=1000, step=1000)
            timeout = st.number_input("timeout", min_value=5, max_value=60, value=60)
            num_populations = st.number_input(
                "num_populations", help="Number of parallel populations to spawn", min_value=1, max_value=3, value=1
            )

        with cols[1]:
            mut_rate = st.number_input("mutation_rate", value=0.05, min_value=0.05, max_value=0.9, step=0.05)
            cross_rate = st.number_input("crossover_rate", value=0.1, min_value=0.05, max_value=0.9, step=0.1)
            elite_rate = st.number_input("elite_rate", value=0.1, min_value=0.05, max_value=0.9, step=0.1)

        text_input = st.text_input(
            label="input_array",
            value="1,2,3,4,5,6,7,8,9,10,11,12,13,14",
            help="Must be a comma-separated string of 14 integers between 1 and 15 inclusive",
        )
        checkbox_val = st.checkbox(" Roll the dice instead! 🎲")
        if not checkbox_val:
            dice_array = text_input.split(",")
        else:
            dice_array = get_roll()
        # Every form must have a submit button.
        kwargs = {
            "dice_array": dice_array,
            "pop_size": pop_size,
            "timeout": timeout,
            "num_populations": num_populations,
            "mut_rate": mut_rate,
            "cross_rate": cross_rate,
            "elite_rate": elite_rate,
        }

        submitted = st.form_submit_button(
            label="Submit to solve!",
            help="Click to submit the puzzle to the solver",
            on_click=get_solve_job,
            kwargs=kwargs,
        )


def get_grid_df_from_dice_array(dice_array):
    # Take every other element from the dice_array (as str) and the text_array, and
    # add them to a new flat list.
    # Make sure the flat list has 28 elements.
    # Then, reshape the list into a 4x7 matrix and then pandas DataFrame
    text_array = ["+", "=", "", "", "-", "=", "", "", ":", "=", "+/-", "x", "=", "+/-"]
    text_positions = [1, 3, 5, 6, 8, 10, 12, 13, 15, 17, 19, 22, 24, 26]
    flat_list = []
    dice_ind = 0
    text_ind = 0
    for i in range(len(dice_array) + len(text_array)):
        if i in text_positions:
            take = text_array[text_ind]
            text_ind += 1
        else:
            take = dice_array[dice_ind]
            dice_ind += 1
        flat_list.append(take)
    grid = np.array(flat_list).reshape(4, 7)
    return pd.DataFrame(grid)


rows = 2
cols = 7
# st.markdown("## Dice values")
# write_grid(cols, rows, st.session_state["dice_array"])


def style_cell(x, correct=False):
    if x.isdigit():
        if not correct:
            return "color: black; background-color: white"
        else:
            return "color: black; background-color: #bbffaa"
    elif x in ("+", "-", ":", "x", "=", "+/-") or x.strip() == "":
        return "color: white; background-color: black"


def style_row(row):
    return [style_cell(c, correct=(st.session_state["correct_equations"][row.name])) for c in row]


def get_df_html():
    dice_array = st.session_state.get("dice_array")
    df = get_grid_df_from_dice_array(dice_array)
    props = [
        {"selector": "table", "props": [("font", "arial")]},
        {
            "selector": "tbody",
            "props": [
                ("font-size", "200%"),
                ("text-align", "center"),
                ("border", "2px solid black"),
            ],
        },
        {
            "selector": "td",
            "props": [("border", "10px solid black"), ("font-weight", "bold")],
        },
    ]
    style_df = df.style.apply(style_row, axis=1).hide(axis=1).hide(axis=0).set_table_styles(props)

    df_html = style_df.to_html()
    return df_html


if "best_score" not in st.session_state:
    st.session_state["best_score"] = 0
if "time_left" not in st.session_state:
    st.session_state["time_left"] = timeout
if "num_tested" not in st.session_state:
    st.session_state["num_tested"] = 0


def millify(n):
    # Use 1 decimal place for numbers >= 1000 and no decimal places for < 1000
    # E.g. 1100 -> 1.1K, 2100000 -> 2.1M
    millnames = ["", "K", "M", "B", "T"]
    n = float(n)
    millidx = max(0, min(len(millnames) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))
    return "{:.1f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])


st.markdown("## Metrics ")
cols = st.columns(3)
with cols[0]:
    st.metric("Score 🎯", int(st.session_state["best_score"]))
with cols[1]:
    st.metric("Seconds left ⌛", st.session_state["time_left"])
with cols[2]:
    st.metric("Solutions tested 🤰", millify(st.session_state["num_tested"]))


st.markdown("## Current solution (Submit to solve👈)")
df_html = get_df_html()
st.markdown(df_html, unsafe_allow_html=True)

cols = st.columns(3)

with cols[1]:
    st.write("")
    poll_button = st.button(
        "Poll results",
        key="poll_button",
        help="👈 Submit to solve first"
        if ("call_id" not in st.session_state) or st.session_state["time_left"] == 0
        else "",
        disabled=("call_id" not in st.session_state) or st.session_state["time_left"] == 0,
        on_click=poll_results,
    )

if all(st.session_state["correct_equations"]):
    st.balloons()
# st.session_state["dice_array"]
# st.session_state["correct_equations"]
# components.html(df_html, width=1000, height=600)

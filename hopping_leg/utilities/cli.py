"""
CLI
===
"""

import argparse
from argparse import RawTextHelpFormatter


def parse(arguments, description="", formatter_class=RawTextHelpFormatter):
    """
    Parse CLI arguments stored as a dictionary.

        {"a|argument": [default value, help message, {extra /argparse add_argument/ argument: "value"}]}

    """
    p = argparse.ArgumentParser(description=description, formatter_class=formatter_class)

    for key in arguments.keys():

        p.add_argument(f"--{key.split('|')[1]}",
                        f"--{key.split('|')[0]}",
                        **arguments[key]
                        )

    return p


def printu(msg, level=0):
    """
    Print underlined messages
    """

    u = {0: "=",
         1: "-",
         2: "~"}

    print(f"\n{msg}\n"+f"{u[level] if level in u.keys() else u[2]}"*len(msg)+"\n")
"""
tal.render
======

Functions for simulating NLOS captured data.

This is a private module, it is recommended to use the command line interface instead of calling these functions directly.

See tal render -h for more information.
"""


def create_nlos_scene(folder_name, args):
    """
    It is recommended to use the command line interface instead of calling this function directly.

    See tal render -h for more information. You probably want to do tal render new <folder_name>
    """
    from tal.render import create
    create.create_nlos_scene(folder_name, args)


def render_nlos_scene(config_path, args):
    """
    It is recommended to use the command line interface instead of calling this function directly.

    See tal render -h for more information. You probably want to do tal render <scene>, or tal render new <folder_name>
    """
    from tal.render import render
    return render.render_nlos_scene(config_path, args)

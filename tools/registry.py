from .basic import TOOLS


def list_tools():
    return list(TOOLS.keys())


def execute_tool(name, arguments=None):
    if arguments is None:
        arguments = {}

    if name not in TOOLS:
        raise ValueError(f"Unknown tool: {name}")

    return TOOLS[name](**arguments)

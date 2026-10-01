import datetime


def get_time():
    """Return current server time."""
    return datetime.datetime.now().isoformat()


TOOLS = {
    "get_time": get_time
}

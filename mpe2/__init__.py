__version__ = "1.1.0"

try:
    import sys

    from farama_notifications import notifications

    if "minigrid" in notifications and __version__ in notifications["minigrid"]:
        print(notifications["mpe2"][__version__], file=sys.stderr)
except Exception:  # nosec
    pass

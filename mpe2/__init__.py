__version__ = "1.1.1"

try:
    import sys

    from farama_notifications import notifications

    if "mpe2" in notifications and __version__ in notifications["mpe2"]:
        print(notifications["mpe2"][__version__], file=sys.stderr)
except Exception:  # nosec
    pass

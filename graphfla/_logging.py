"""Package logging setup.

graphfla emits progress messages through the standard :mod:`logging` module,
under the ``graphfla`` logger. By library convention nothing is shown by
default (a :class:`~logging.NullHandler` is attached, so an application that
does not configure logging sees no output).

Building a landscape with ``verbose=True`` calls :func:`enable_verbose_logging`,
which lazily attaches a stderr handler at ``INFO`` level so the gated progress
messages appear — preserving the old print-based experience. The per-instance
``if landscape.verbose:`` gates still decide *whether* a message is produced, so
a ``verbose=False`` landscape stays silent regardless of global logger state.

Applications wanting structured output can ignore ``verbose`` and configure the
``graphfla`` logger themselves (handlers, level, formatting) as usual.
"""

import logging

logger = logging.getLogger("graphfla")
logger.addHandler(logging.NullHandler())

_verbose_handler = None


def enable_verbose_logging():
    """Idempotently route ``graphfla`` INFO logs to stderr (used by verbose=True).

    Attaches a single stderr handler (message-only formatting, to mirror the
    previous ``print`` output) and raises the logger level to INFO if it is
    currently higher. Safe to call repeatedly.
    """
    global _verbose_handler
    if _verbose_handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        _verbose_handler = handler
        logger.addHandler(handler)
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)

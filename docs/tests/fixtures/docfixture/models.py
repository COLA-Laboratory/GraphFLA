class Base:
    def inherited(self, amount: int = 1) -> int:
        """An inherited public method.

        Parameters
        ----------
        amount : int, optional
            INHERITED_PARAMETER.

        Returns
        -------
        int
            INHERITED_RETURN.
        """
        return amount


class Box(Base):
    """A small class with state and inherited behavior.

    Parameters
    ----------
    size : int, optional
        INITIAL_SIZE.

    Attributes
    ----------
    size : int
        Stored box size.

    Notes
    -----
    CLASS_NOTE.

    Examples
    --------
    >>> 1 + 1
    2
    """

    def __init__(self, size: int = 4):
        """Initialize the box.

        Notes
        -----
        CONSTRUCTOR_NOTE.
        """
        self.size = size

    @property
    def total(self) -> int:
        """PROPERTY_SENTINEL for the current total."""
        return self.size

    def method(self, value: int = 2):
        """A direct public method.

        Parameters
        ----------
        value : int, optional
            DIRECT_PARAMETER.
        """
        return value

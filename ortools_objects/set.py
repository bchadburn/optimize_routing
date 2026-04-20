from ortools_objects.component import ORComponent


class ORSet(ORComponent):
    """Set object upon which all indexed components of the math model are based.

    Kwargs:
        name (str): Name of the set for string representation
        doc (str): Doc string of the set for string representation
        initialize (list): Initial values of the set in list form

    Example use: I have a constraint and need to create for a bunch of distribution sites. I need a number of items producted per site, a parameter of minimum products producted
     per site, a constraint linking variable items to minimum items produced, and a set of distribution sites
    to use for Indexed sites. Note that all other components depend on this set.

    model.s_distribution_sites = ORSet(name='foo', doc='foo', initialize=['site0', 'site1', 'site2', ...])

    Now that the set is create, I can use it to create an indexed parameter (see param.py), indexed constraint (see constraint.py),
    and all other model components.
    """

    def __init__(self, *args, **kwds):
        kwds.setdefault("ctype", "set")

        self._data = kwds.pop("initialize")
        if not isinstance(self._data, list):
            raise TypeError("ORSet initialization must be a list")

        ORComponent.__init__(self, *args, **kwds)

        self._constructed = True

    def __len__(self):
        return len(self._data)

    def __contains__(self, idx):
        return idx in self._data

    def __getitem__(self, idx):
        return self._data[idx]

    def cross(self, *args) -> list:
        """Returns a list of tuples containing the product of the current set (self) and any other sets passed to this function.
        Mainly used by indexed components to generate all possible tuples of a combination of sets, but can be used externally
        for other purposes (debugging, etc)

        Returns:
            list: List of tuples containing the product of the current set (self) and any other sets passed to this function.
        """
        from itertools import product

        return list(product(self._data, *[arg._data for arg in args]))

    def __call__(self):
        return self._data

    def __iter__(self):
        yield from self._data

    __mul__ = cross

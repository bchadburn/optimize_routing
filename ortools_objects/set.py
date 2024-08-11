from numbers import Number
from typing import Union

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

    Now that the set is created, I can use it to create an indexed parameter (see param.py), indexed constraint (see constraint.py),
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

        # Sub-function to take care of case when multi-index set is compound and multi-dimensional
        def convert(data):
            result = []
            for item in data:
                if isinstance(item, tuple):
                    result.extend(convert(item))
                else:
                    result.append(item)
            return tuple(result)
        
        return [
            convert(item)
            for item in list(product(self._data, *[arg._data for arg in args]))
        ]

    # Returns the data in the set if called as function
    def __call__(self):
        return self._data

    # Used to iteratively yield elements from the set for the purposes of constraint calls
    def __iter__(self):
        yield from self._data

    def previous(self, element: Union[str, Number]):
        """Returns the previous element. If its the first element in the set, the last element is returned.
        
        Args:
            element (str, Number): Element in the set for which the previous element is to be found
        
        Returns:
            str, Number: Element in the set that is previous to the element passed in as an argument
        """
        idx = self._data.index(element)
        return self._data[idx - 1]
    
    def next(self, element: Union[str, Number]):
        """Returns the next element in the set. If its the last element in the set, None is returned
        
        Args:
            element (str, Number): Element in the set
        
        Returns: (str, Number): Next element in the set"""
    
        idx = self._data.index(element)
        if idx == len(self._data) - 1:
            return None
        return self._data[(idx + 1)]
        
    __mul__ = cross

from numbers import Number
from typing import Union

from ortools_objects.component import ORComponent


class ORSet(ORComponent):
    """
    A set object representing a collection of elements in an ORTools optimization model.
    Sets define the index sets over which other components, such as parameters, variables, and constraints, are defined. 
    This class provides a way to create and manage sets within the model.

    Kwargs:
        name (str): A descriptive name for the set, used for string representation.
        doc (str): A documentation string providing additional details about the set.
        initialize (list): A list containing the initial elements of the set.
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
        """
        Computes the Cartesian product of the current set with one or more other sets.

        This method generates a list of tuples representing all possible combinations of elements from the current set and the sets provided as arguments. The Cartesian product is a fundamental operation in set theory and is commonly used in optimization models to create index sets for indexed components.

        Args:
            *args: One or more ORSet objects to be combined with the current set.

        Returns:
            list: A list of tuples, where each tuple contains one element from the current set and one element from each of the provided sets, representing all possible combinations.

        Example:
            Suppose you have a set of products and a set of distribution sites, and you want to create an indexed parameter that stores the demand for each product at each distribution site. You can use the `cross` method to generate the index set for this parameter:

            model.s_products = ORSet(name='products', initialize=['prod1', 'prod2', 'prod3'])
            model.s_distribution_sites = ORSet(name='distribution_sites', initialize=['site1', 'site2', 'site3'])

            index_set = model.s_products.cross(model.s_distribution_sites)
            # index_set = [('prod1', 'site1'), ('prod1', 'site2'), ('prod1', 'site3'), ('prod2', 'site1'), ...]

            model.p_demand = IndexedORParam(index_set, name='demand', ...)

            In this example, the `index_set` contains all possible combinations of products and distribution sites, which can be used to define the indexed parameter `p_demand`.
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

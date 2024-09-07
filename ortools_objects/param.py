
import logging
from numbers import Number
from typing import Union

from ortools_objects.component import ORComponent
from ortools_objects.indexed_component import IndexedComponent


class IndexedORParam(IndexedComponent):
    """Indexed parameter object for an ORTools model.

    Args:
        A number of ORSet objects that will be crossed together in IndexedComponent init dunder to create the index set.

    Kwargs:
        doc (str): A doc string that can be used in the string representation of the parameter
        name (str): The name of the parameter that will be used to name the dictionary entries
        initialize (dict): Dictionary with indices as the keys and values of the param as values
        default (Number, Optional): Default value of the parameter for missing indices. Defaults to 0.

    Example use: I have minimum items produced for given distribution sites. I want constraints in the model to be able to access this without
    directly needed to access shifting model architectures. I can create, first, a set of distribution sites.

    model.s_distribution_sites = ORSet(name='distribution_sites',doc='foo',initialize=['site0', 'site1', 'site2', ...])

    Now that I have a set of distribution sites, I want to store in the model the minimum items produced:

    model.p_minimum_items_produced = IndexedORParam(model.s_distribution_sites, name='foo', doc='foo', initialize={'site0': 50, 'site1': 10, 'site2': 5})

    Now that it is a part of the model structure, I can use it in constraint rules (see constraint objects for more details):

    def example_constraint(model, site):
        model.v_items_produced[site] >= model.p_minimum_items_produced[site]
    """

    def __init__(self, *args, **kwds):
        # Set the component type to parameters
        kwds.setdefault("ctype", "param")

        # Validate args passed to component
        self._validate_args(args)

        # Populate default value
        self._default_val = kwds.pop("default", 0)
        if not isinstance(self._default_val, Number):
            raise TypeError("Default value for parameter must be a number")

        # Initialize the value. Will raise error if no data
        self._initialize = kwds.pop("initialize")
        if not isinstance(self._initialize, dict):
            raise TypeError(
                "Indexed parameter data must come in the form of a dictionary"
            )

        # Initialize indexed components
        IndexedComponent.__init__(self, *args, **kwds)

        # Take initialize and transfer it to data
        if self._initialize is not None:
            self._data = self._initialize
        else:
            raise ValueError(
                "Initialize kwarg should be a dictionary and is mandatory to construct"
            )

        # Validate the indices to make sure they are a part of the index set
        self._validate_data_indices(self._data)

        self._constructed = True

    def _validate_args(self, args):
        IndexedComponent._validate_args(self, args)

    def _getitem_when_not_present(self, index: Union[str, tuple]) -> float:
        """If there is a default value provided, retrieve it for a value that does not exist at index.
        If no default value, return a ValueError.

        Args:
            index (str | tuple): Index for which the value is missing

        Returns:
            Number: Default value if relevant
        """
        if self._default_val is not None:
            return self._default_val
        else:
            return IndexedComponent._getitem_when_not_present(self, index)

    def log_parameter_values(self, logger: logging.Logger):
        if not logger:
            return
        for index in self:
            if self[index] != 0:
                logger.info(
                    f"Value of {self._name} at indices {self._index_name}: {index}  is {self[index]}."
                )


class ScalarORParam(ORComponent):
    """Scalar OR parameter containing a single value

    Kwargs:
        doc (str): A doc string that can be used in the string representation of the parameter
        name (str): The name of the parameter.
        initialize (Number, Optional): Value of the scalar parameter

    Example use: Let's say that the cost for some component of my objective is fixed per unit  (e.g. $15/product). I could store
    this single value in a parameter such that is accessible from a callable used in a constraint:

    model.p_cost_of_unit = ScalarORParam(name='foo', doc='foo', initialize=17)

    Now, it is possible to use the cost of unit in any constraint if so desired. Most likely, this would be used in an objective function callable:

    def objective_function(model):
        return model.p_cost_of_unit * sum(v_use[index] for index in model.set)
    """

    def __init__(self, *args, **kwds):
        kwds.setdefault("ctype", ScalarORParam)
        self._data = kwds.pop("initialize", 0)
        if not isinstance(self._data, Number):
            raise TypeError(
                "Initialize argument should have some type of number for component type ScalarORParam"
            )
        ORComponent.__init__(self, *args, **kwds)

    # Returns dimension
    def __dim__(self):
        return 1

    def __getitem__(self, index):
        if index is not None:
            raise KeyError(f"Index access for scalar param {self._name} must be None.")
        return self._data

    # Returns the data if called as function
    def __call__(self):
        return self._data


import logging
from numbers import Number
from typing import Union

from ortools_objects.component import ORComponent
from ortools_objects.indexed_component import IndexedComponent


class IndexedORParam(IndexedComponent):
    """
    An indexed parameter object for an ORTools optimization model.
    This class represents a parameter whose values are indexed over one or more sets. 
    Indexed parameters are useful for storing data or values that vary across different 
    combinations of set elements. They can be used in constraints, objective functions, and other model components.

    Args:
        *sets: One or more ORSet objects that will be used to create the index set for the parameter.

    Kwargs:
        doc (str): A documentation string providing a description of the parameter.
        name (str): The name of the parameter, which will be used as the dictionary key for accessing its values.
        initialize (dict): A dictionary containing the initial values of the parameter, where the keys are the index 
        combinations and the values are the corresponding parameter values.
        default (Number, optional): The default value to be used for any missing index combinations. Defaults to 0.

    Example:
        Suppose you have a set of distribution sites, and you want to store the minimum number of items that must be 
        produced at each site. First, create the set of distribution sites:

        model.s_distribution_sites = ORSet(name='distribution_sites', doc='Set of distribution sites', initialize=['site0', 'site1', 'site2'])

        Then, create an indexed parameter to store the minimum production values:

        model.p_minimum_items_produced = IndexedORParam(
            model.s_distribution_sites,
            name='minimum_items_produced',
            doc='Minimum number of items to be produced at each distribution site',
            initialize={'site0': 50, 'site1': 10, 'site2': 5}
        )

        You can now use this indexed parameter in constraints or other model components:

        def production_constraint(model, site):
            return model.v_items_produced[site] >= model.p_minimum_items_produced[site]
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
    """
    A scalar parameter object for an ORTools optimization model.
    This class represents a parameter that holds a single, scalar value. 
    Scalar parameters are useful for storing constant values that are used throughout the model, 
    such as costs, penalties, or other fixed parameters.

    Kwargs:
        doc (str): A documentation string providing a description of the parameter.
        name (str): The name of the parameter, which will be used to access its value.
        initialize (Number, optional): The initial value of the scalar parameter.

    Example:
        Suppose you want to model the cost per unit of a product, which is a fixed value of $10. You can create a scalar parameter to store this value:

        model.p_cost_per_unit = ScalarORParam(
            name='cost_per_unit',
            doc='Cost per unit of the product',
            initialize=10
        )

        You can then use this scalar parameter in constraints, objective functions, or other model components:

        def objective_function(model):
            return model.p_cost_per_unit * sum(model.v_units_produced[index] for index in model.s_products)

        In this example, the objective function calculates the total cost by multiplying the cost per unit (`model.p_cost_per_unit`) by the sum of units produced for each product (`model.v_units_produced`).
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

import logging
from typing import Callable, Union

from ortools_objects.component import ORComponent
from ortools_objects.model import ORToolsCPModel


class ORObjective(ORComponent):
    """
    This class represents the objective function of an optimization model.
    The objective function is a mathematical expression that defines the quantity 
    to be minimized or maximized in the optimization problem. This class provides a convenient way to specify the objective function and its associated properties.

    Args:
        name (str): A descriptive name for the objective function.
        doc (str): A documentation string providing additional details about the objective function.
        rule (Callable): A callable function that takes the model object as input and returns the mathematical expression representing the objective function.
        sense (str, optional): The optimization direction, either 'minimize' or 'maximize'. Defaults to 'minimize'.

    Example:
        Suppose you want to minimize the total cost of products across multiple sites. You have an `ORSet` object representing the sites, 
        and a variable `v_product_cost` indexed over the sites. You can define the objective function as follows:

        def objective_function(model):
            return sum(model.v_product_cost[site] for site in model.s_sites)

        model.objective = ORObjective(
            name='total_cost',
            doc='Minimize the total cost of products across all sites',
            rule=objective_function,
            sense='minimize'
        )
 
    Note:
        The attribute name of the `ORObjective` instance in the model object must be 'objective' for the `ORToolsCPModel` to function correctly.
    """

    def __init__(self, *args, **kwds):
        kwds.setdefault("ctype", "objective")

        self._rule = kwds.pop("rule")
        assert isinstance(
            self._rule, Callable
        ), "Rule argument must be a callable function"
        self._sense = kwds.pop("sense", "minimize")
        assert (
            self._sense == "minimize" or self._sense == "maximize"
        ), "Sense argument must be in set minimize, maximize"

        ORComponent.__init__(self, *args, **kwds)

    def construct(
        self,
        model_wrapper: ORToolsCPModel,
        logger: Union[logging.Logger, None] = None
    ):
        """Add the objective function into the model using the rule

        Args:
            model_wrapper (ORToolsCPModel): Total model object
            logger (logging.Logger, Optional): Logger to log information about the objective
        """
        original_vars = len(list(model_wrapper.mathopt_model.variables()))
        if self._sense == "minimize":
            model_wrapper.mathopt_model.minimize(self._rule(model_wrapper))
        else:
            model_wrapper.mathopt_model.maximize(self._rule(model_wrapper))
        new_vars = len(list(model_wrapper.mathopt_model.variables()))
        if logger:
            logger.info(
                f"Added objective to model. Added {new_vars-original_vars} variables to model."
            )
        self._constructed = True

import logging
from collections.abc import Callable

from ortools.linear_solver import pywraplp

from ortools_objects.component import ORComponent
from ortools_objects.model import ORToolsCPModel


class ORObjective(ORComponent):
    """Set object upon which all indexed components of the math model are based.

    Kwargs:
        name (str): Name of the set for string representation
        doc (str): Doc string of the set for string representation
        rule (Callable): Callable that returns an expression in ORTools, taking in full model object as input
        sense (str, Optional): Whether the function should be minimized or maximized. Defaults to 'minimize'

    Example use: I want to charge $1/product for cost optimization. I have a set of sites in the form of an ORSet object,
    a variable representing cost use indexed over those sites, and I want to minimize overall cost.

    def objective_function(model):
        return 1 * sum(model.v_product_cost[site] for site in model.s_sites)

    Now that I have my callable function, I can create the objective function object:
    model.objective = ORObjective(name='objective', doc='objective', rule=objective_function, sense='minimize')

    Note that the attribute name of the ORObjective in the model object must be objective in order for ORToolsCPModel to function correctly.
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
        model: ORToolsCPModel,
        solver: pywraplp.Solver,
        logger: logging.Logger = None,
    ):
        """Add the objective function into the model using the rule

        Args:
            model (ORToolsCPModel): Total model object
            solver (pywraplp.Solver): Solver for which to add the objective
            logger (logging.Logger, Optional): Logger to log information about the objective
        """
        original_vars = solver.NumVariables()
        if self._sense == "minimize":
            solver.Minimize(self._rule(model))
        else:
            solver.Maximize(self._rule(model))
        new_vars = solver.NumVariables()
        if logger:
            logger.info(
                f"Added objective to model. Added {new_vars-original_vars} variables to model."
            )
        self._constructed = True

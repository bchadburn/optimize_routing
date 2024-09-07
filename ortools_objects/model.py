from collections import defaultdict
from datetime import timedelta
from typing import Any

from ortools.math_opt.python import mathopt

from ortools_objects.component import ORComponent
from ortools_objects.component_decorators import ComponentDecorator


def factory():
    return defaultdict(factory)


class ORToolsCPModel:
    """Constraint programming modeling object to hold all sets, parameters, variables, and constraints for the model.
    Main use is to take assigned attributes of the model and construct a model instance using the rest of the model objects.
    The class is written in a way that does not add variables/constraints to the problem until the construct model instance is called.

    The model contains all objects to be used and constructed and is the core of the model. It can contain sets (from set.py),
    parameters (from param.py), variables (from var.py), constraints (from constraint.py), and a single objective function
    (from objective.py) to make models.

    In general, it is best to create sets, followed by parameters, followed by variables, followed by constraints, followed by an
    objective function. To see details of each of these components, proceed to files listed above.

    The only model specific note is the general workflow: (0) create model, (1) create sets, (2) create parameters, (3) create variables, (4) create
    constraints, (5) create objective, (6) call construct_model() method, and (7) call solve_model() method.
    """

    def __init__(self, **kwds):
        """Initializes the raw components of the model, mainly
        empty attributes and sets/parameters imported from the data model.

        Kwargs:
            solver (str, Optional): Solver to use. Current solver is HiGHS.
            max_time (Number, Optional): Time limit of solver in sections. Defaults to 30.
            rel_gap (Number, Optional): Gap for MIP that is allowed. Defaults to 0.01 (1%).
            log (bool, Optional): Declares whether or not to print detailed log. Defaults to True.
            callback (Callable, Optional): How to print the log information. Defaults to print.
            logger (logging.Logger): Logging
            Other KWARGS (Optional): Can be used in any way to pass information to the model to be used by constraints in the form of model_config[key]
        """
        # Create the solver
        self.mathopt_model: mathopt.Model = mathopt.Model(name="MathOpt Model")
        # Objective and model constructed attributes
        self.model_constructed = False
        
        self.solve_parameters = mathopt.SolveParameters()
        # Set the model options using kwargs
        if kwds.pop("solver_log", True):
            self.solve_parameters.enable_output = True
        self.solve_parameters.random_seed = kwds.pop("seed", 0)
        self.solve_parameters.relative_gap_tolerance = kwds.pop("rel_gap", 0.01)
        self.solve_parameters.time_limit = timedelta(seconds=kwds.pop("max_time", 1200))

        self.logger = kwds.pop("logger", None)
        # Save remaining kwargs to model config
        self.model_config = kwds

    def __getattr__(self, component_name: str) -> Any:
        """Takes in a component type with some arguments. If the component name is valid (in the factor),
        then it returns a Component decorator object, taking the component reference as an argument,
        that can be used to add the component to the model."""
        from ortools_objects.component_factory import ComponentFactory
        
        if str(component_name) in ComponentFactory():
            return ComponentDecorator(
                self, ComponentFactory.retrieve_component(component_name)
            )
            
        raise AttributeError(
            f"Attribute {component_name} is not a valid component type"
            
        )
        
    def validate_model(self):
        """Validate the model to make sure that two model objects don't have the same name"""
        from collections import Counter

        names = Counter(
            [
                model_obj._name
                for model_obj in self.__dict__.values()
                if isinstance(model_obj, ORComponent)
            ]
        )
        for name, count in names.items():
            if count > 1:
                raise ValueError(
                    f"Name {name} is used by more than two model objects. Check your model declaration and remove any duplicates."
                )

    def construct_model(self):
        """Construct the model using all variable, constraint, and objective attributes."""
        self.validate_model()

        for value in self.__dict__.values():
            if (
                isinstance(value, ORComponent)
                and value.ctype == "var"
                and not value.is_constructed()
            ):
                value.construct(self, self.logger)

        for value in self.__dict__.values():
            if (
                isinstance(value, ORComponent)
                and value.ctype == "constraint"
                and not value.is_constructed()
            ):
                value.construct(self, self.logger)

        # Construct the objective
        for value in self.__dict__.values():
            if (
                isinstance(value, ORComponent)
                and value.ctype == "objective"
                and not value.is_constructed()
            ):
                value.construct(self, self.logger)
                
        if self.logger:
            self.logger.debug(self.mathopt_model.export_model())
        self.model_constructed = True

    def process_results(self):
        """Take all variable results in attribute dictionary and process the results"""
        for value in self.__dict__.values():
            if isinstance(value, ORComponent) and value.ctype == "var":
                value.process_result(self.result, self.logger if self.logger else None)

    def solve_model(self):
        """Solve generated math model. Model must be constructed prior to call or
        Attribute error will be raised.

        Returns:
            Union[None, Number]: None if model not feasible. Else return the status number
        """
        if self.model_constructed:
            for value in self.__dict__.values():
                if isinstance(value, ORComponent) and value.ctype == "var":
                    value._solved = False
            # Solve the model
            self.result = mathopt.solve(
                self.mathopt_model,
                mathopt.SolverType.HIGHS,
                params=self.solve_parameters
            )
            
            self.status = self.result.termination.reason

            if self.status == mathopt.TerminationReason.OPTIMAL:
                self.process_results()
                return self.status
            elif self.status == mathopt.TerminationReason.FEASIBLE:
                self.process_results()
                if self.logger:
                    self.logger.warning(
                        "Feasible but not optimal solution found. Try giving the optimizer or a little longer to solve or increasing the relative gap option."
                    )
                return self.status
            else:
                if self.logger:
                    self.logger.warning(
                        "Feasible solution not found. Check model LP file for possible infeasibility or run with slack variables on."
                    )
                return None
        else:
            raise AttributeError(
                "Model has not yet been constructed. First run construct_model successfully."
            )

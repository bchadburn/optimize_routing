import ortools
import sympy
from sympy import Add, Max
from sympy.core.numbers import Number

import utils.log as log
from ortools_objects.model import ORToolsCPModel

logger = log.get_logger("SCIPSolver")


def convert_expr_to_ortools(
    expr: sympy.core.expr.Expr,
    model: ORToolsCPModel,
    time_index: int = None,
) -> ortools.linear_solver.pywraplp.Variable:
    """Takes in a sympy expression representing an objective function
    and translates it into a solver expression for ORTools.

    Args:
        expr (sympy.core.expr.Expr): Expression in sympy to be translated
        solver (ortools.linear_solver.pywraplp.Solver): Solver in ORTools to add the expression/variables to
        model (ORToolsCPModel): Model containing variables to be used
        time_index (int, Optional): Time index for pressure expressions, defaults to None (not needed for cost expr)

    Returns:
        ortools.linear_solver.pywraplp.Variable: _description_
    """

    def return_args(expr, model, time_index):
        list_args = []
        if hasattr(expr, "args"):
            for arg in expr.args:
                add_arg = convert_expr_to_ortools(arg, model, time_index)
                list_args.append(add_arg)
        return list_args

    def args_from_sum(expr, model, time_index):
        add_args = return_args(expr, model, time_index)

        add_expr = 0
        for add_args in add_args:
            add_expr += add_args
        return add_expr

    def args_from_mult(expr, model, time_index):
        mul_args = return_args(expr, model, time_index)

        mul_expr = 1
        if len(mul_args) == 2:
            for mul_arg in mul_args:
                mul_expr *= mul_arg
        else:
            raise NotImplementedError("Multiplication of 3+ vars is not supported.")
        return mul_expr

    def define_max(solver, str_expr):
        str_expr = "'" + str_expr.replace("-", "_minus_") + "'"
        aux_var = solver.NumVar(0.0, solver.infinity(), str_expr)
        return aux_var

    def set_max_constraints(args, aux_var, model, time_index):
        for arg in args.args:
            max_var = convert_expr_to_ortools(arg, model, time_index)
            model.lp_solver.Add(aux_var >= max_var)

    if isinstance(expr, Number):
        return float(expr)
    
    elif expr.func == Add:
        add_expr = args_from_sum(expr, model, time_index)
        return add_expr

    elif expr.is_Mul:
        mul_expr = args_from_mult(expr, model, time_index)
        return mul_expr

    elif expr.func == Max:
        aux_var = define_max(str(expr), model)
        set_max_constraints(expr, aux_var, model, time_index)
        return aux_var
    # Example of checking for specific class in order to return variable from model
    # elif isinstance(expr, VariableClassExampls):
    #     return model.v_example[time_index, expr.site_name]
    else:
        raise NotImplementedError(f"{type(expr)} not supported")

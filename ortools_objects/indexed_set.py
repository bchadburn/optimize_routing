from ortools_objects.indexed_component import IndexedComponent


class IndexedORSet(IndexedComponent):
    """
    An indexed set object for an ORTools optimization model.

    This class represents a set whose elements are indexed over one or more other sets. Indexed sets are primarily used in constraints to iterate over specific portions of sets for certain index combinations. They should not be used as initializers for indexed components, as this will raise an error.

    Args:
        *sets: One or more ORSet objects that will be used to create the index set for the indexed set.

    Kwargs:
        name (str): A descriptive name for the indexed set.
        doc (str): A documentation string providing additional details about the indexed set.
        initialize (dict): A dictionary containing the initial elements of the indexed set, where the keys are the index tuples, and the values are lists or sets of elements.

    Example:
        Suppose you have a set of sites and a set of time periods, and during preprocessing, you generate a series of piecewise segments for each site and time period combination. However, the number of piecewise segments may vary for different site/time period combinations. To sum over all piecewise segments for a given site and time period in a constraint, you can use an indexed set as follows:

        model.s_sites = ORSet(name='sites', initialize=['site1', 'site2', 'site3'])
        model.s_time_periods = ORSet(name='time_periods', initialize=[0, 1, 2])

        model.s_piecewise_segments = IndexedORSet(
            model.s_time_periods, model.s_sites,
            name='piecewise_segments',
            doc='Set of piecewise segments for each site and time period',
            initialize={(0, 'site1'): [0, 1, 2], (0, 'site2'): [0, 1], (1, 'site1'): [0, 1], ...}
        )

        def production_constraint(model, time_period, site):
            return model.v_actual_production[time_period, site] == sum(
                model.v_piecewise_production[time_period, site, piecewise_idx]
                for piecewise_idx in model.s_piecewise_segments[time_period, site]
            )

    In this example, the indexed set `s_piecewise_segments` allows the constraint to sum over the appropriate piecewise segments for each site and time period combination, even though the number of segments may vary.
    """

    def __init__(self, *args, **kwds):
        kwds.setdefault("ctype", "set")

        # self._validate_args(args)

        self._initialize = kwds.pop("initialize")

        IndexedComponent.__init__(self, *args, **kwds)
        if isinstance(self._initialize, dict):
            self._data = self._initialize
        else:
            raise TypeError(
                f"Indexed ORSet {self._name} initialize keyword must contain dictionary"
            )

    def _validate_args(self, args):
        IndexedComponent._validate_args(args)

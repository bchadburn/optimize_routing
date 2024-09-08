class ORComponent:
    """
    The base component object for an ORTools optimization program. This class serves as the foundation for all other 
    components in the ORTools optimization model, except for the model component itself. All other classes representing components, 
    such as sets, parameters, variables, and constraints, should be derived from this base class.
    
    When initializing an instance of a subclass of `ORComponent`, it should only contain a docstring, a component type, and a name. 
    The component type and name are typically defined in the subclass itself. If desired, you can override the `__str__` method 
    (the string dunder) to provide a custom string representation for the component.
    
    Note: This class should never be instantiated directly. Instead, it should be extended by other classes representing specific types of 
    components in the optimization model.
    """


    def __init__(self, **kwds):
        self._ctype = kwds.pop("ctype", None)
        self.doc = kwds.pop("doc", None)
        self._name = kwds.pop("name", str(type(self).__name__))
        # If initializing from decorator, may have a rule object. But if the rule is not None, there is an issue.
        if "rule" in kwds:
            if kwds["rule"] is not None:
                raise AttributeError(
                    "Rule attribute should only exist for constraint objects."
                )
            else:
                kwds.pop("rule")
        if kwds:
            raise ValueError(
                "Unexpected keyword options found while constructing '%s':\n\t%s"
                % (type(self).__name__, ",".join(sorted(kwds.keys())))
            )


        if self._ctype is None:
            raise ValueError(
                "Must specify a component type for class %s." % (type(self).__name__,)
            )


        self._constructed = False
        self._parent = None


    @property
    def ctype(self):
        return self._ctype


    def type(self):
        return self._ctype


    def is_constructed(self):
        return self._constructed


    def __str__(self):
        return self._name + ((": " + self.doc) if self.doc else "")

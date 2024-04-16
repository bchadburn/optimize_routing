
class ORComponent:
    """Base component object for ORTools optimization program. All other classes, save for the model component, are based on this component.
    By the time this object is initialized (from a subclass), it should only contain a docstring, a component type, and a name.
    If you wish, you can modify the string dunder to create a different string representation of every component.

    This one should never be used directly.
    """

    def __init__(self, **kwds):
        self._ctype = kwds.pop("ctype", None)
        self.doc = kwds.pop("doc", None)
        self._name = kwds.pop("name", str(type(self).__name__))
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

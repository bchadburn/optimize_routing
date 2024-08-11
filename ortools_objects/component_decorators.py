class _GenericComponentDecorator:
    """Decorator to add the component passed through the decorator cascade to the model tree.
    This addds the component to the model and allows it to be constructed with the model construction.
    """
    
    def __init__(self, model, component, *args, **kwargs):
        self._model = model
        self._component = component 
        self._args = args
        self._kwargs = kwargs
        
    def __call__(self, rule=None):
        model_prefix = self._component["model_prefix"]
        component_object = self._component["component_object"]
        if hasattr(self._model, f"{model_prefix}{rule.__name__}"):
            raise AttributeError(f"Component {rule.__name__} already exists in model.")
        setattr(
            self._model, 
            f"{model_prefix}{rule.__name__}",
            component_object(*self._args, rule=rule, **self._kwargs)
        )
        
class ComponentDecorator:
    """Wraps the generic component decorator, which preserves the model object and the component type.
    To abstrat away all of the arguments not related to the model or component objects, another layer of abstractions is provided for the rest of the arguments.
    """
    def __init__(self, model, component): 
        self._model = model
        self._component = component
        
    def __call__(self, *args, **kwds):
        # Due to the fact that we just created a decorator, we cannot use the rule directly. Rather we need first to determine what the arguments for the rule are
        # This is accomplished through this dunder
        return _GenericComponentDecorator(self._model, self._component, *args, **kwds)
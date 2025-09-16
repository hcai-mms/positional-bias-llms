class ModelLibrary(dict):

    def __init__(self, *arg, **kw):
      super(ModelLibrary, self).__init__(*arg, **kw)

    def register(self, name=None, model_class=None):
        if model_class is not None:
            self[name] = model_class
            return model_class
        
        def _decorator(cls):
            self.register(name=name.lower(), model_class=cls)
            return cls
        return _decorator

model_library = ModelLibrary()
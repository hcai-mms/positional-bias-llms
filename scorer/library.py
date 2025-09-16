class ScorerLibrary(dict):

    def __init__(self, *arg, **kw):
      super(ScorerLibrary, self).__init__(*arg, **kw)

    def register(self, name=None, scorer_class=None):
        if scorer_class is not None:
            self[name] = scorer_class
            return
        
        def _decorator(cls):
            self.register(name=name, scorer_class=cls)
        return _decorator

scorer_library = ScorerLibrary()
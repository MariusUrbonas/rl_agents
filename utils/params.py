class Params():

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__dict__[key.upper()] = value

    def add_params(self, **kwargs):
    	  for key, value in kwargs.items():
            self.__dict__[key.upper()] = value

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__
    
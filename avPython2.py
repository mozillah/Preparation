

class A(object):
    """ SImple way to create class and it's objects"""
    def __init__(self):
        pass

# 2nd Method

class B(object):
    def __new__(cls, *args, **kwargs):
        return super(B, cls).__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):




class D(type):
    def __call__(cls, *args, **kwargs):
        instance = super(D, cls).__call__()


#
# class B(object):
#     """Using a __call__ method to create object """
#     def __call__(self, *args, **kwargs):
#         return  super(B, self).__call__()
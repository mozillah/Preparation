class A (object):
    def __init__(self):
        pass

class B(object):
    """
    That __call__() method in turn invokes the following:

    __new__()
__init__()
    """
    def __new__(cls, *args, **kwargs):
        return super(B,cls).__new__(cls,*args,**kwargs)
    def __init__(self, *args, **kwargs):
         super(B,self).__init__(*args,**kwargs)

    # def __call__(cls, *args, **kwargs):
    #     print("hello world")
    #     return super(B, cls).__call__(cls, *args, **kwargs)




# __new__ is the first step in instance construction, invoked before __init__

# __new__ simply allocates memory for the object. The instance variables of an object needs memory to hold it, and this is what the step __new__ would do.

# __init__ initialize the internal variables of the object to specific values (could be default).

class C(object):
    def __new__(cls, *args, **kwargs):
        return super(C,cls).__new__(cls, *args, **kwargs)













# First Method to create class

class A(object):
    def __init__(self):
        pass
class B(object):
    def __new__(cls, *args, **kwargs):
        return super(B, cls).__new__(cls, *args, **kwargs)

    def __init__(self,*args,**kwargs):
        return super(B, self).__init__(*args,**kwargs)



class C(object):
    def __call__(cls, *args, **kwargs):
        return super(C, cls).__call__(cls, *args, **kwargs)



class Meta(type):
    def __new__(cls, *args, **kwargs):
        return super(Meta,cls).__new__(cls, *args, **kwargs)

    def __init__(self,*args,**kwargs):
        super(Meta,self).__init__(*args,**kwargs)


class DM(metaclass=Meta):
    pass

class MetaNew(type):
    def __call__(cls, *args, **kwargs):
        instance =  super(MetaNew,cls).__call__( *args, **kwargs)
        return instance
    def __init__(cls,name,base,attr):
        super(MetaNew,cls).__init__(name,base,attr)


class D(metaclass=MetaNew):
    pass



class F(type):
    def __call__(cls, *args, **kwargs):
        instance =super(F,cls).__call__(*args, **kwargs)
        return instance
    def __init__(self,name,base,attr):
        super(F, self).__init__(name,base,attr)

class FM(metaclass=F):
    pass

d = D()
print(d)


dm= DM()
print(dm)

fm = FM()
print(fm)

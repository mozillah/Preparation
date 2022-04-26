

class Test(object):

    #class Variable
    myvar = 222
    def __init__(self):
        self.num = 1234
        self.__numa = 1234

    def get(self):
        return self.__numa

    def set(self,numa):
        self.__numa = numa

    @property
    def methoda(self):
        print("I am method")

    @staticmethod
    def methodb():
        print("I am static")

    def __str__(self):
        return  "Object of test"

    def __repr__(self):
        return "Object of test for console.."

    def classvaracess(cls):
        print("Class variable access {}".format(cls.myvar))






if __name__ == "__main__":
    obj = Test()
    print(obj)
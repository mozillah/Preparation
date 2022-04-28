import csv


class Item:

    all =[]
    pay_rate = 0.8
    def __init__(self,name:str,price:float,quantity:int):
        assert price >=0 ,f"Price  {price} is not greater than or equal to zero"
        assert quantity >=0 ,f"Quantity  {quantity} is not greater than or equal to  zero"
        self.name =name
        self.price = price
        self.quantity = quantity
        Item.all.append(self)


    @property
    def name(self):
        return self.name
    @name.setter
    def name(self,value):
        self.__name =value
    def calculate_total_price(self):
        return self.price * self.quantity

    def apply_discount(self):
        self.price= self.price * self.pay_rate
        return self.price

    @classmethod
    def instantiate_from_csv(cls):
        with open("Items.csv",'r') as f:
            reader = csv.DictReader(f)
            items =list(reader)

        for item in items:
            Item(name=item.get("name"),
                 quantity=int(item.get("quantity") ),
                 price=int(item.get("price")),
                 )

    def __repr__(self):
        return f" name {self.name} price {self.price},quantity{self.quantity}"


class Phone(Item):
    all=[

    ]

    def __init__(self, name: str, price: float, quantity: int,broken_phones:int):
        super().__init__(name,price,quantity)
        assert broken_phones >=0 ,f"Quantity  {broken_phones} is not greater than or equal to  zero"


        self.broken_phones =broken_phones
        Phone.all.append(self)



    pass

















class PyItem(object):
    item_rate =0.8
    all =[]
    def __init__(self,name:str,price:int,quantity:int):
        assert price >=0, f"Price {price} is not greater than or equal to zero"
        assert quantity >=0,f"Quantity {quantity} is not greater than or equal to zero"
        self.name = name
        self.price = price
        self.quantity = quantity
        Item.all.append(self)
    def calculate_Item(self,x,y):
        return  x*y

# Use generally for factory pattern
# Factory methods return class objects ( similar to a constructor ) for different use cases.
#  method that is bound to the class not the object of the class
# it can be used for instantiating the objects
    @classmethod
    def instantiate_from_csv(cls):
        with open("Items.csv") as f:
            reader = csv.DictReader(f)
            items =list(reader)
        for item in items:
            Item(item.get("name"),float(item.get("price")),int(item.get("quantity")))
            # print(item)

    @classmethod
    def claass(cls):
        print(cls.__dict__)


    @staticmethod
    def is_integer(num):
        if isinstance(num,float):
            # print(num.__init__())
            return num.is_integer()
        elif isinstance(num,int):
            return True
        else:
            return False

    def __repr__(self):
        return f"Item ({self.name} , {self.price}, {self.quantity})"

    def calculate_total_price(self):
        return self.price*self.quantity






class Phone(Item):
    all =[]
    def __init__(self,name:str,price:float,quantity:int,broken_phones:int):
        assert price >=0, f"Price {price} is not greater than or equal to zero"
        assert quantity >=0,f"Quantity {quantity} is not greater than or equal to zero"
        assert quantity >=0,f"Quantity {quantity} is not greater than or equal to zero"
        super(Phone, self).__init__(name,price,quantity)
        self.__broken_phones = broken_phones
        Phone.all.append(self)


if __name__ == '__main__':
    item1 = PyItem(name="cake1",price=10,quantity=10)
    item1.item_rate = 0.1
    item2 = PyItem(name="cake2",price=20,quantity=30)
    # item3 = PyItem(name="cake3",price=0,quantity=-1)
    print(item1.calculate_total_price())
    print(item2.calculate_total_price())
    print(item2.item_rate)
    print(item2.__dict__)
    print(Item.all)
    PyItem.instantiate_from_csv()
    print(PyItem.is_integer(10.1))
    phone1 = Phone("jsPhone",10,1)
    phone1.broken_phones =1
    phone2 =Phone("new",20,20)
    # print(item.name)
    # print(item.calculate_Item(1, 2))

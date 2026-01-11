#print ("Hello world!")
#print ("\n")

# This is a comment explaining the code.

# Numbers
#x = 10
#y = 3.5

# Text (string)
#name = "Istiak"

# Boolean
#is_hacker = True

#print(x, y, name, is_hacker)

# Python figures out the data type automatically.

#a = 5               # integer
#b = 2.5             # float
#c = "CyberSec"      # string
#d = True            # boolean

#print(type(a))
#print(type(b))
#print(type(c))
#print(type(d)) 
#print ("\n")
# Python is dynamically typed, so you can change the type of a variable.

#x = 10
#y = 3

#print(x + y)   # addition
#print(x - y)   # subtraction
#print(x * y)   # multiplication
#print(x / y)   # division (float result)
#print(x // y)  # floor division (int)
#print(x % y)   # modulus (remainder)
#print(x ** y)  # power (10^3)
#print("\n")

#text = "Python is fun!"

#print(text.lower())     # lowercase
#print(text.upper())     # uppercase
#print(text.replace("fun", "powerful"))
#print(text[0])          # first character
#print(text[-1])         # last character
#print(text[0:6])        # slicing (0 to 5)

#name = input("Enter your name: ")
#print("Hello", name, "welcome to Python!")

#age = int(input("Enter your age: "))

#if age >= 18:
#    print("You are an adult.")
#else:
#    print("You are underage.")
#print("\n")

# While loop example
#i = 1
#while i <= 5:
#    print("Count:", i)
#    i = i + 1
#print("Loop finished!")
# The above code demonstrates a simple while loop that counts from 1 to 5.

# For loop example
#for i in range(5):   # 0 to 4
#    print("Loop:", i)

#for char in "SPY*WarriorBD_TheSpyder.apk":
#    print(char)

fruits = ["apple", "banana", "cherry"]

print(fruits[0])       # first item
print(fruits[-1])      # last item

fruits.append("orange") # add item
fruits.remove("banana") # remove item
print(fruits)

for fruit in fruits:
    print("I like", fruit)



def greet(name):
    return f"Hello {name}, welcome to Python!"

print(greet("Istiak"))
print(greet("Hacker"))

print("Simple Calculator")
print("Enter two numbers:")

num1 = int(input("First number: "))
num2 = int(input("Second number: "))

print("Choose an operation:")
print("1. Addition")
print("2. Subtraction")
print("3. Multiplication")
print("4. Division")
choice = input("Enter your choice (1/2/3/4): ")
if choice == '1':
    result = num1 + num2
    print("The Sum is:", result)
elif choice == '2':
    result = num1 - num2
    print("The Sub is:", result)
elif choice == '3':
    result = num1 * num2
    print("The Multiplication is:", result)
elif choice == '4':
    if num2 != 0:
        result = num1 / num2
        print("The Division is:", result)
    else:
        print("Error: Division by zero is not allowed.")
else:
    print("Invalid choice. Please select a valid operation.")

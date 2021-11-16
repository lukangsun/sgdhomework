import math
def f(x):
    return 0.25*((x**-1)+3)+1/(math.log(1-x))

print(f(0.001))
#minor helping functions

#checks if "place"-th bit is set in the integer x 
def bitAt(x, place):
    n = place-1
    y = 2**n
    return (x&y)>>n

#[a,b] -> [0,1]
def transformToZeroOne(value, lowerBound, upperBound):
    return (value - lowerBound)/(upperBound - lowerBound)

#[0,1] -> [a,b]
def transformFromZeroOne(value, lowerBound, upperBound):
    return value*(upperBound - lowerBound) + lowerBound
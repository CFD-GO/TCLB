import sys
import h5py
import numpy as np

try:
    file1 = sys.argv[1]
    file2 = sys.argv[2]
except IndexError:
    print("Please supply two files for comparison...")
    sys.exit(1)

if len(sys.argv) == 4:
    tol = float(sys.argv[3])
else:
    tol = 1e-6

# Read in both files:
data1 = h5py.File(file1,'r')
data2 = h5py.File(file2,'r')

# Compare both files have the same fields:
keys1 = np.array(list(data1.keys()))
keys2 = np.array(list(data2.keys()))

# Compare difference of all fields
if len(np.setdiff1d(keys1,keys2)) != 0:
    print("The two files have different fields...")
    print(keys1)
    print(keys2)
    sys.exit(1)

# Compare all the fields values
for i in keys1:
    tmp1 = np.array(data1[i])
    tmp2 = np.array(data2[i])
    
    diff = np.max(abs(tmp2 - tmp1))
    if diff > tol:
        print("Difference in field %s is %f which is greater than tolerance %f" % \
               (i, diff, tol) )
        sys.exit(1)

print("No differences greater than tolerance %f found..." % (tol))
sys.exit(0)

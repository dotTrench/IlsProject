from collections import Counter
import bisect

list1 = [3, 2, 5, 1, 6]
list2 = ["T", "F", "F", "F", "T"]

list1, list2 = zip(*sorted(zip(list1, list2)))

print(list1)
print()
print(list2)

print(list1[:1])
print(list1[1:])

c = Counter(list2)

print(c)

# for k, v in c


def find_gt(a, x):
    i = bisect.bisect_right(a, x)
    if i != len(a):
        return a[i]
    raise ValueError



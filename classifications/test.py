from collections import Counter

test = ['A', 'B', 'A', 'A', 'C']

counts = Counter(test)

print(counts)

for key, c in counts.items():
    print(c)
    print(len(test))
    print('\n')


x = [1,2,3,4,5,6]
for i in range(len(x)):
    print(i)
n = int(input())
dict = {}
for i in range(n):
    a, b = input().split()
    a = int(a)
    b = int(b)

    if dict.get(a) is not None:
        dict[a] = max(dict.get(a), int(b))
    else:
        dict[a] = int(b)
print(dict.keys())

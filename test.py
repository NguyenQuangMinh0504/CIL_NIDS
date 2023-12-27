N = 50
table = []
for i in range(N):
    table.append([0] * N)

n, m = [int(i) for i in input().split()]
for i in range(1, n + 1):
    for j in range(1, m + 1):
        table[i][j] = 
PS = []
for i in range(n):
    row = [0] * (m + 1)
    PS.append(row)
for i in range(n):
    for j in range(1, m + 1):
        PS[i][j] = table[i][j] + PS[i][j-1]
Q = int(input())

for _ in range(Q):
    r1, c1, r2, c2 = [int(i) - 1 for i in input().split()]
    print("result is: ")
    print(PS[r2][c2] - PS[r2][c1 - 1] - PS[r1 - 1][c2] + PS[r1 - 1][c1 - 1])

print(table)
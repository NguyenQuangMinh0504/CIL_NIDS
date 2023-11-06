import matplotlib.pyplot as plt
rmm_foster = [96.4, 88.8, 82.07, 75.9, 74.76, 71.67, 68.94, 66.7, 65.42, 63.92, 62.47, 60.85, 59.43, 58.0, 57.2, 55.69, 54.09, 53.76, 51.97, 50.7]
coil = [97.2, 89.0, 82.13, 74.85, 73.24, 68.8, 66.49, 59.72, 57.91, 55.66, 53.27, 52.02, 49.28, 46.61, 44.23, 40.56, 38.52, 36.71, 35.73, 34.87]
finetune = [94.2, 45.5, 30.47, 21.5, 19.32, 17.13, 13.83, 12.52, 11.6, 10.58, 8.98, 6.82, 7.06, 7.34, 6.45, 6.2, 5.86, 5.18, 4.95, 5.14]
foster = [96.4, 91.4, 82.53, 75.75, 73.24, 69.33, 65.03, 61.7, 59.18, 58.1, 57.09, 56.55, 55.74, 55.41, 53.59, 51.24, 51.58, 50.91, 49.25, 48.87]
memo = [96.2, 91.4, 81.47, 77.05, 76.04, 72.73, 70.94, 68.22, 66.96, 65.96, 64.35, 62.98, 61.85, 61.44, 59.75, 57.95, 56.98, 56.37, 56.16, 55.34]

# print(len(my_memo))

my_memo_len = [i for i in range(0, 55, 5)]
x = [i for i in range(0, 100, 5)]
plt.plot(x, rmm_foster, color="red", label="rmm_foster")
plt.plot(x, coil, color="green", label="coil")
plt.plot(x, finetune, color="pink", label="finetune")
plt.plot(x, foster, color="black", label="foster")
plt.plot(x, memo, color="orange", label="memo")
# plt.plot(my_memo_len, my_memo, color="grey")
plt.legend(loc='upper right')
plt.show()

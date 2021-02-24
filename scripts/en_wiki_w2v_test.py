maximum = 1
i = 0

print("Reading file...")
# with open("skip_s300.txt", "r") as f:
#     for i in range(maximum):
#         print(f'line {i}: {f.readline()}')
# with open("enwiki_20180420_300d.txt", "r") as f:
#     for i in range(maximum):
#         print(f'line {i}: {f.readline()}')
with open("enwiki_20180420_300d.txt", "r") as f:
    for line in f:
        print(f'{line}')

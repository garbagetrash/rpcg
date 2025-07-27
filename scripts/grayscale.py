print("const greyscale: [[u8; 4]; 256] = [")
for i in range(256):
    print(f"    [{i}, {i}, {i}, 255],")
print("];")

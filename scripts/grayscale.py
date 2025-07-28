print("pub const greyscale: [[u8; 4]; 256] = [")
for i in range(256):
    if i <= 127:
        # Waterline
        print(f"    [{i}, {i}, {127+i}, 255],")
    elif i > 127 and i % 8 == 0:
        print(f"    [255, 255, 255, 255],")
    else:
        print(f"    [{2*(i-128)}, {i}, {2*(i-128)}, 255],")
print("];")

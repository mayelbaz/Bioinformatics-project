def convert_to_int(word):
    return ''.join(str(ord(c)) for c in word)


if __name__ == '__main__':
    print(convert_to_int("0\0"))
    print(convert_to_int("0\\1"))
    print(convert_to_int("1\\0"))
    print(convert_to_int("0\\2"))
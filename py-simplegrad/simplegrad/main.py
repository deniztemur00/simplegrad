from simplegrad import Space 


def main():
    space = Space(3.2)
    space2 = Space(4.8)
    print(space)
    print(space2)
    print("operator +: ", space + space2)
    print("operator +: ", space * space2)
    print("operator **: ", space**3.2)
    print("operator -: ", -space)
    print("operator -: ", space - space2)
    print("operator /: ", space / space2)
    print("operator /: ", space2 * space)


if __name__ == "__main__":
    main()

import time


def read_file(file_name):
    global error_file_count
    lines = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
    return lines


def main():
    # read_file("G:\\UB\\5th Sem\\Individual Project\\Data\\QS-OCR-Large\\imagesa\\a\\a\\a\\aaa08d00\\2072197187.txt")
    print("ok")


if __name__ == '__main__':
    start_time = time.time()
    print(start_time)
    main()
    ennd_time = time.time()
    execusion_time = ennd_time - start_time
    print("--- %s seconds ---" % (time.time() - start_time))

# count documents for test train and val per category   X
# normalize the words counts and line counts with documents number per category  X
# discard all the empty files  X
# discrad handwritten class  X
# seaborn  X
# paper reading
# sklear clusterring

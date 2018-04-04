import tools

if __name__ == "__main__":
    data = tools.load_data_scilab('../data/doryo/day_0109_270.dat')
    data = tools.convert(data)
    data = tools.threshold(data)